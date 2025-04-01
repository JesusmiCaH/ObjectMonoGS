from gaussian_model import GaussianModel
import torch.nn as nn
import torch
import numpy as np
from utils import getWorld2View2, RGB2SH, distCUDA2, inverse_sigmoid
from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, InstancePointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p

class RigidModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._point_ids = torch.empty(0,device = 'cuda')  # (N, 1)
        self.instance_pose = torch.empty(0,device = 'cuda')  # (Frame, instancce, 4, 4)
    
    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, inst_id_map=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        instance_id = inst_id_map
        if depthmap is not None:
            rgb = rgb_raw.astype(np.uint8)
            depth = depthmap.astype(np.float32)

        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = rgb_raw.astype(np.uint8)
            depth = depth_raw.astype(np.float32)

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, instance_id, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, instance_id, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()

        u = np.arange(0, depth.shape[0], 1)  # height
        v = np.arange(0, depth.shape[1], 1)  # width
        uu, vv = np.meshgrid(u, v)
        uu = uu.flatten()
        vv = vv.flatten()
        dd = depth.flatten()
        color_flatten = rgb.reshape(-1, 3)
        ins_id_flatten = instance_id.flatten()

        dd_valid = dd > 0
        uu = uu[dd_valid]
        vv = vv[dd_valid]
        dd = dd[dd_valid]
        color_flatten = color_flatten[dd_valid]
        ins_id_flatten = ins_id_flatten[dd_valid]

        z = dd
        x = (vv - cam.cx) * z / cam.fx
        y = (uu - cam.cy) * z / cam.fy

        points_C = np.stack((x, y, z), axis=1)  # (N, 3)
        N = points_C.shape[0]
        points_C_homo = np.concatenate((points_C, np.ones([N,1])), axis = 1) # (N, 4)
        # Transform 2 World Coordinate
        C2W = np.linalg.inv(W2C)
        points_W_homo = (C2W @ points_C_homo.T).T
        points_W = points_W_homo[:, :3]  # (N, 3)

        # Downsampling
        indices = np.random.choice(
            points_W.shape[0], 
            size=int(points_W.shape[0] / downsample_factor), 
            replace=False
        )
        points_W = points_W[indices]
        color_flatten = color_flatten[indices]
        ins_id_flatten = ins_id_flatten[indices]

        new_xyz = points_W
        new_rgb = color_flatten
        new_ins_id = ins_id_flatten

        pcd = InstancePointCloud(
            points=new_xyz, colors=new_rgb, instance_ids=new_ins_id, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        fused_ins_id = torch.from_numpy(np.asarray(pcd.instance_ids)).float().cuda()
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, fused_ins_id, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, fused_ins_id, scales, rots, opacities, kf_id
    ):
        # 下一步改这边！！！！！！！！
        # 这边！！！！
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, inst_id_map=None
    ):
        fused_point_cloud, features, fused_ins_id, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, inst_id_map=inst_id_map)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, fused_ins_id, scales, rots, opacities, kf_id
        )
