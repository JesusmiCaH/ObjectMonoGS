from .gaussian_model import GaussianModel
import torch.nn as nn
import torch
import numpy as np
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
from gaussian_splatting.utils.general_utils import (
    build_rotation,
    quat_mult,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, InstancePointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p

from mpl_toolkits.mplot3d import Axes3D

class DynamicGaussianModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ins_ids = torch.empty(0, device='cuda', dtype=torch.long)  # (N)
        self._current_frame = 0
        self.seen_ins_ids = torch.empty(0, device='cuda', dtype=torch.long)

    @property
    def get_ins_ids(self):  
        return self.ins_ids # (N)
    @property
    def get_ins_means(self):
        return self._instance_means # (Frame, instance, 3)
    @property
    def get_ins_quats(self):
        return self._instance_quats # (Frame, instance, 4)
    @property
    def get_xyz(self):
        # rot_per_gaussian = build_rotation(self._instance_quats[self._current_frame, self.ins_ids]) # (N, 3, 3)
        # mean_per_gaussian = self._instance_means[self._current_frame, self.ins_ids] # (N, 3)
        # with torch.no_grad():
        #     rot_per_gaussian[self.ins_ids==0] = torch.eye(3).cuda()
        #     mean_per_gaussian[self.ins_ids==0] = torch.zeros(3).cuda()
        # transformed_points = torch.bmm(rot_per_gaussian, self._xyz.unsqueeze(2)).squeeze(2) + mean_per_gaussian # (N, 3)
        # return transformed_points
        return self._xyz
    
    @property
    def get_rotation(self):
        return self._rotation
        # quats_per_gaussian = self._instance_quats[self._current_frame, self.ins_ids] # (N, 4)
        # with torch.no_grad():
        #     quats_per_gaussian[self.ins_ids==0] = torch.tensor([0, 0, 0, 1], dtype=torch.float).cuda()
        # return quat_mult(quats_per_gaussian, self._rotation) # (N, 4)
    
    def set_frame(self, frame_id):
        # Set the current frame to the specified frame ID
        self._current_frame = frame_id

    # def densification_newframe(self, new_cam):
    #     (pcd, _, ins_id, _, _, _) = self.create_pcd_from_image(
    #         new_cam,
    #         init=False,
    #         scale=2.0,
    #         depthmap=new_cam.depth,
    #     )

    #     n_inst = self._instance_means.shape[1]
    #     new_ins_means = torch.ones((1, n_inst, 3)).cuda()
    #     new_ins_quats = self._instance_quats[self._current_frame-1].unsqueeze(0)
        
    #     for seen_instance in self.seen_ins_ids:
    #         if seen_instance == 0:
    #             continue
    #         new_ins_means[0, :, :] = pcd[ins_id == seen_instance].mean(dim=0)
            
    #     pose_d = {
    #         "instance_means": new_ins_means,
    #         "instance_quats": new_ins_quats,
    #     }
    #     optimizable_poses = self.cat_rigidpose_to_optimizer(pose_d, mode = "new_frame")
    #     self._instance_means = optimizable_poses["instance_means"]
    #     self._instance_quats = optimizable_poses["instance_quats"]
        
    def training_setup(self, training_args):
        super().training_setup(training_args)
        # l_rigid = [
        #     {
        #         "params": [self._instance_means],
        #         "lr": training_args.inst_mean_lr,
        #         "name": "instance_means",
        #     },
        #     {
        #         "params": [self._instance_quats],
        #         "lr": training_args.inst_quat_lr,
        #         "name": "instance_quats",
        #     }
        # ]
        # self.rigid_optimizer = torch.optim.Adam(l_rigid)
    
    def cat_rigidpose_to_optimizer(self, poses_dict, mode = "new_frame"):
        """
        Similar to cat_tensors_to_optimizer but specifically for rigid poses.
        Concatenates new instance poses to the existing ones and updates the optimizer state.
        Should be leveraged when new frame is added to the model.
        Args:
            poses_dict: Dictionary with keys 'instance_means' and 'instance_quats' containing new pose tensors
        
        Returns:
            Dictionary with updated parameter tensors
        """
        optimizable_poses = {}
        # Determine the dimension along which to concatenate
        cat_dim = 0 if mode == "new_frame" else 1

        for group in self.rigid_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = poses_dict[group["name"]]
            stored_state = self.rigid_optimizer.state.get(group["params"][0], None)
            
            if stored_state is not None:
                # Extend optimizer state
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=cat_dim
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=cat_dim,
                )

                # Replace parameter in optimizer
                del self.rigid_optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=cat_dim
                    ).requires_grad_(True)
                )
                self.rigid_optimizer.state[group["params"][0]] = stored_state

                optimizable_poses[group["name"]] = group["params"][0]
            else:
                # If no state exists yet, simply concatenate the tensors
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=cat_dim
                    ).requires_grad_(True)
                )
                optimizable_poses[group["name"]] = group["params"][0]

        return optimizable_poses
    
    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

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

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        W2C = getWorld2View2(cam.R, cam.T).detach().cpu().numpy()

        # --------------------------------------------------------------- #

        u = np.arange(depth.shape[0])  # height
        v = np.arange(depth.shape[1])  # width
        vv, uu = np.meshgrid(v, u)
        uu = uu.flatten()
        vv = vv.flatten()
        dd = depth.flatten()
        color_flatten = rgb[uu, vv] / 255.0  # (N, 3)
        ins_id_flatten = cam.segment_map.flatten()  # (N)

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

        new_xyz = points_W
        new_rgb = color_flatten
        new_ins_id = ins_id_flatten
        

        # Downsampling
        if downsample_factor > 1.0:
            sampled = np.random.rand(N) < (1.0/downsample_factor)

            new_xyz = new_xyz[sampled]
            new_rgb = new_rgb[sampled]
            new_ins_id = new_ins_id[sampled]


        pcd = InstancePointCloud(
            points=new_xyz, colors=new_rgb, instance_ids=new_ins_id, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        fused_ins_id = torch.from_numpy(np.asarray(pcd.instance_ids)).cuda()
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

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_ins_id,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        super().densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_ids,
            new_n_obs=new_n_obs,
        )

        self.ins_ids = torch.cat((self.ins_ids, new_ins_id), dim=0).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)     # (N*mask_size, 3)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_ins_id = self.ins_ids[selected_pts_mask].repeat(N)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_ins_id,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_ins_id = self.ins_ids[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_ins_id,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def prune_points(self, mask):
        super().prune_points(mask)
        valid_points_mask = ~mask
        self.ins_ids = self.ins_ids[valid_points_mask]

    def filter_unseen_ins_id(self, segment_data):
        collected_ids = segment_data.unique()
        unseen_mask = ~torch.isin(collected_ids, self.seen_ins_ids)
        return collected_ids[unseen_mask]

    def extend_from_pcd(
        self, fused_point_cloud, features, fused_ins_id, scales, rots, opacities, kf_id
    ):
        # 下一步改这边！！！！！！！！
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        unseen_ins_ids = self.filter_unseen_ins_id(fused_ins_id)
        
        self.seen_ins_ids = torch.cat((self.seen_ins_ids, unseen_ins_ids), dim = 0)

        for unseen_instance in unseen_ins_ids:
            print("Unseen Instance ID: ", unseen_instance)
            if unseen_instance == 0:
                continue
            
            xyz_dist, _, _ = knn_points(self._xyz[None,...], new_xyz[fused_ins_id==unseen_instance][None,...])

            self.ins_ids[(xyz_dist[0,:,0]<0.01) & (self.ins_ids==0)] = self.seen_ins_ids.shape[0]

        # with torch.no_grad():
        #     new_xyz[fused_ins_id == unseen_instance] -= new_ins_means[0, 0, :]

        id2idx = { v:i for i,v in enumerate(self.seen_ins_ids.tolist())}
        # print(id2idx.items())
        new_ins_id = torch.tensor([id2idx[i] for i in fused_ins_id.tolist()]).cuda()
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_ins_id,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, fused_ins_id, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, fused_ins_id, scales, rots, opacities, kf_id
        )
