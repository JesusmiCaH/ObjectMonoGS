import numpy as np
import torch
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    # print("kankanTau", tau.norm(), "kankanthreshold", converged_threshold)
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def compound_pose(camera):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    
    W2V = getWorld2View2(camera.R, camera.T)
    W2V_T = (SE3_exp(tau) @ W2V).transpose(0, 1)
        
    return W2V_T

def compound_projection(camera):
    W2V_T = compound_pose(camera)
    Proj = camera.projection_matrix
    
    return W2V_T.unsqueeze(0).bmm(Proj.unsqueeze(0)).squeeze(0)

def back_projection(camera, point_2d_ndc, depth, P_W2f_t):
    """
    Back project 2D points to 3D space using the camera's projection matrix.
    Args:
        camera: Camera object containing intrinsic and extrinsic parameters.
        point_2d_ndc: 2D points in normalized device coordinates (NDC). Shape: (N, 2).
        depth: Depth values for the 2D points. Shape: (N,).
        P_W2f_t: Projection matrix from world to camera coordinates. Shape: (4, 4).
    """

    H,W = camera.depth.shape
    
    u = (point_2d_ndc[:, 0] + 1.0) * (W - 1) / 2.0
    v = (point_2d_ndc[:, 1] + 1.0) * (H - 1) / 2.0
    
    u_proj = (u - camera.cx) * depth / camera.fx
    v_proj = (v - camera.cy) * depth / camera.fy
    
    point_3d_proj_homo = torch.stack(
        (u_proj, v_proj, depth, depth.clone().fill_(1)), dim=1
    )
    
    # Invert the projection matrix
    P_f2W_t = P_W2f_t.inverse()

    point_3d = point_3d_proj_homo @ P_f2W_t
    
    return point_3d[:, :3]