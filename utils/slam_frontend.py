import time

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, compound_pose, back_projection
from utils.slam_utils import get_loss_tracking, get_median_depth
from romatch import roma_outdoor, roma_indoor
import time

from simple_knn._C import distCUDA2
from gaussian_splatting.utils.sh_utils import RGB2SH


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_pose_tracking = 0
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        # Load matcher
        self.roma_model = roma_indoor(device=self.device)
        self.roma_model.upsample_preds = False
        self.roma_model.symmetric = False

        self.seen_instances = set()

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False
    
    
    def get_warps(self, viewpointA, viewpointB):
        imA = viewpointA.original_image.cpu().numpy().transpose(1, 2, 0)
        imA = (imA * 255).astype(np.uint8)
        imA = Image.fromarray(imA)
        imB = viewpointB.original_image.cpu().numpy().transpose(1, 2, 0)
        imB = (imB * 255).astype(np.uint8)
        imB = Image.fromarray(imB)

        with torch.no_grad():
            warp, certainty_warp = self.roma_model.match(
                imA, imB, device=self.device
            )
        return warp, certainty_warp

    def get_matches(self, viewpointA, viewpointB, num_matches=15000):
        warp, certainty_warp = self.get_warps(
            viewpointA, viewpointB
        )
        warp = warp.reshape(-1, 4)  # H*W x 4
        certainty_warp = certainty_warp.reshape(-1)  # H*W
        good_samples = torch.multinomial(certainty_warp, num_matches, replacement=False)
        kpts_A, kpts_B = warp[good_samples].split(2, dim=1)
        return kpts_A, kpts_B
    

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                # "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "lr": 0.001,
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                # "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "lr": 0.001,
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        kptsA_ndc, kptsB_ndc = self.get_matches(
            viewpoint, prev
        )

        kptsA_x = torch.round((kptsA_ndc[:,0] + 1) * (viewpoint.image_width - 1) / 2).long()
        kptsA_y = torch.round((kptsA_ndc[:,1] + 1) * (viewpoint.image_height - 1) / 2).long()
        kptsB_x = torch.round((kptsB_ndc[:,0] + 1) * (viewpoint.image_width - 1) / 2).long()
        kptsB_y = torch.round((kptsB_ndc[:,1] + 1) * (viewpoint.image_height - 1) / 2).long()

        depth_A = torch.from_numpy(viewpoint.depth).to(self.device).float()
        depth_B = torch.from_numpy(prev.depth).to(self.device).float()
        
        dd_valid = (depth_A[kptsA_y, kptsA_x] > 0) & (depth_B[kptsB_y, kptsB_x] > 0) 
        
        kptsA_ndc = kptsA_ndc[dd_valid]
        kptsB_ndc = kptsB_ndc[dd_valid]

        depth_A = depth_A[kptsA_y[dd_valid], kptsA_x[dd_valid]]
        depth_B = depth_B[kptsB_y[dd_valid], kptsB_x[dd_valid]]

        P3d_B = back_projection(prev, kptsB_ndc, depth_B, prev.world_view_transform)
            
        for tracking_itr in range(self.tracking_itr_num):
            P3d_A = back_projection(viewpoint, kptsA_ndc, depth_A, compound_pose(viewpoint))

            pose_optimizer.zero_grad()
            loss_tracking = torch.abs(P3d_A - P3d_B).mean()
            

            # --------------------------------------------------

            # render_pkg = render(
            #     viewpoint, self.gaussians, self.pipeline_params, self.background
            # )
            # image, depth, opacity = (
            #     render_pkg["render"],
            #     render_pkg["depth"],
            #     render_pkg["opacity"],
            # )
            # image = image[0]
            # depth = depth[0]
            # opacity = opacity[0]
            # pose_optimizer.zero_grad()
            # loss_tracking = get_loss_tracking(
            #     self.config, image, depth, opacity, viewpoint, "static"
            # )

            # --------------------------------------------------
            
            # print("loss_tracking", loss_tracking.item())
            loss_tracking.backward()
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint, converged_threshold=1e-4)

            # if tracking_itr % 10 == 0:
            #     self.q_main2vis.put(
            #         gui_utils.GaussianPacket(
            #             current_frame=viewpoint,
            #             gtcolor=viewpoint.original_image,
            #             gtdepth=viewpoint.depth
            #             if not self.monocular
            #             else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            #         )
            #     )
            
            if converged:
                print("iteration", tracking_itr, "converged")
                break
        
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )

        self.median_depth = get_median_depth(depth[0], opacity[0])
        return render_pkg

    def has_new_instance(self, cur_frame_idx, viewpoint):
        segment_map = torch.from_numpy(viewpoint.segment_map).cuda()
        unseen_ins_ids = self.gaussians.filter_unseen_ins_id(segment_map)
        return unseen_ins_ids.numel()!=0

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def init_with_corr(self, current_window, num_matches=10000):
        H, W = self.cameras[current_window[0]].original_image.shape[1:3]
        # 1) for each frame in the current window, find 3 closest camera poses
        camera_pose_all = torch.stack(
            [self.cameras[kf_idx].world_view_transform.flatten() for kf_idx in current_window], axis=0
        ) # N x (4*4)
        camera_dists = torch.cdist(camera_pose_all, camera_pose_all, p = 2) # N x N
        camera_dists.fill_diagonal_(float("inf"))

        Neighbor_num = min(3, len(current_window) - 1)
        knn_dists, knn_indices = torch.topk(camera_dists, Neighbor_num, largest=False) # N x 3

        opt_params = []
        for camidx in current_window:
            opt_params.append({
                "params": [self.cameras[camidx].cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(camidx),
            })
            opt_params.append({
                "params": [self.cameras[camidx].cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(camidx),
            })
        self.cam_pose_optimizer = torch.optim.Adam(opt_params)

        all_xyzs, all_feat_dc, all_feat_rest, all_opacity, all_scaling, all_rotation = [], [], [], [], [], []

        all_kf_IDs, all_n_obs = [], []


        for cam_idx in range(len(current_window)):
            viewpoint = self.cameras[current_window[cam_idx]]

            # 2) for each frame, find their warp_all and confidence_all among the 3 closest frames
            warp_all, confidence_all = [], []
            for nn in knn_indices[cam_idx]:
                cam_nn = self.cameras[current_window[nn]]
                warp, confidence = self.get_warps(viewpoint, cam_nn)
                warp_all.append(warp)
                confidence_all.append(confidence)
            warp_all = torch.stack(warp_all, dim=0)  # nn x H x W x 4
            warp_all = warp_all.reshape(warp_all.shape[0], -1, 4) # nn x (H*W) x 4
            confidence_all = torch.stack(confidence_all, dim=0)  # nn x H x W
            
            # # Biggest tolerance
            # confidence_max, confidence_idx = torch.max(confidence_all, dim=0)  # H x W
            # confidence_max = confidence_max.reshape(-1) # (H*W)
            # confidence_max[confidence_max > 0.9] = 1
            # good_samples = torch.multinomial(confidence_max, num_matches, replacement=False)

            # Smallest tolerance
            confidence_min, confidence_idx = torch.min(confidence_all, dim=0)  # H x W
            confidence_min = confidence_min.reshape(-1) # (H*W)
            confidence_min[confidence_min > 0.9] = 1
            good_samples = torch.multinomial(confidence_min, num_matches, replacement=False)
            print("全站点", (confidence_min > 0.9).sum(), "confidence_min > 0.9")

            kpts_cur = warp_all[0][good_samples, :2]  # num_matches x 2
            kpts_cur_x = ((kpts_cur[:, 0] + 1) * (W - 1) / 2).long()
            kpts_cur_y = ((kpts_cur[:, 1] + 1) * (H - 1) / 2).long()

            kpts_cur_color = viewpoint.original_image[:, kpts_cur_y, kpts_cur_x].transpose(0, 1) # num_matches x 3
            # 3) for each frame, find the best warp and confidence
            x_points, errors = [], []
            for nn, view_idx in enumerate(knn_indices[cam_idx]):
                
                kpts_nn = warp_all[nn][good_samples, 2:]
                # 所有跟cur相关的实际上都是无效计算，记得要改
                
                proj_A = torch.stack([viewpoint.full_proj_transform]*num_matches, dim=0) # num_matches x 4 x 4
                proj_B = torch.stack([self.cameras[current_window[view_idx]].full_proj_transform]*num_matches, dim=0) # num_matches x 4 x 4

                P_A_col0 = proj_A[:,:,0]
                P_A_col1 = proj_A[:,:,1]
                P_A_col2 = proj_A[:,:,2]
                P_B_col0 = proj_B[:,:,0]
                P_B_col1 = proj_B[:,:,1]
                P_B_col2 = proj_B[:,:,2]

                A1 = P_A_col0 - kpts_cur[:, 0].view(-1,1) * P_A_col2  # col0_a - u_a * col2_a
                A2 = P_A_col1 - kpts_cur[:, 1].view(-1,1) * P_A_col2  # col1_a - v_a * col2_a
                A3 = P_B_col0 - kpts_nn[:, 0].view(-1,1) * P_B_col2  # col0_b - u_b * col2_b
                A4 = P_B_col1 - kpts_nn[:, 1].view(-1,1) * P_B_col2  # col1_b - v_b * col2_b

                A = torch.stack([A1, A2, A3, A4], dim=1)  # num_matches x 4 x 4

                A_real = A[:,:,:3]
                b = -A[:,:,3]  # Ax = b
                X_xyz = torch.linalg.lstsq(A_real, b).solution  # num_matches x 3
                x_points.append(X_xyz)

                X_xyz_homo = torch.cat([X_xyz, torch.ones_like(X_xyz[:, :1])], dim=1)  # num_matches x 4
                back_proj_cur = (X_xyz_homo.unsqueeze(1) @ proj_A).squeeze(1)
                back_proj_cur = back_proj_cur / (0.0001 + back_proj_cur[:, [3]])
                err_A = torch.abs(back_proj_cur[:,:2] - kpts_cur).sum(dim=1) # num_matches

                back_proj_nn = (X_xyz_homo.unsqueeze(1) @ proj_B).squeeze(1)
                back_proj_nn = back_proj_nn / (0.0001 + back_proj_nn[:, [3]])
                err_B = torch.abs(back_proj_nn[:,:2] - kpts_nn).sum(dim=1) # num_matches
                err_val = torch.max(err_A, err_B)   # num_matches
                errors.append(err_val)

            
            x_points = torch.stack(x_points, axis=0)
            print("大道偏移", x_points.mean(dim=0).shape, x_points.var(dim=0).mean(dim=0))
            errors = torch.stack(errors, axis=0)

            real_nn_idx = torch.argmin(errors, dim=0)  # num_matches
            aux_idx = torch.arange(real_nn_idx.shape[0]).long()  # num_matches

            selected_x_points = x_points[real_nn_idx, aux_idx, :] # num_matches x 3
            
            all_xyzs.append(selected_x_points)

            all_feat_dc.append(RGB2SH(kpts_cur_color).unsqueeze(1))
            all_feat_rest.append(torch.stack([self.gaussians._features_rest[-1].clone().detach() * 0.0] * len(real_nn_idx))) # num_matches x 15(feat_rest)
            
            good_point_mask = (errors[real_nn_idx, aux_idx] < 0.05)[:, None]   # You can edit this value as tolerance
            
            dist2cam = torch.linalg.norm(selected_x_points - viewpoint.camera_center, dim=1, keepdim=True).squeeze(1) # num_matches x 1
            print("汪汪汪", dist2cam.min())

            all_opacity.append(torch.ones(len(real_nn_idx),1).cuda() * good_point_mask) # num_matches

            dist2 = torch.clamp_min(distCUDA2(selected_x_points), 0.000001) * self.config["Dataset"]["point_size"] * 2
            
            dist2[dist2 > 0.01] = 0.01
            all_scaling.append(torch.log(torch.sqrt(dist2)).unsqueeze(1).repeat(1,3)) # num_matches x 3
            print("喵喵", torch.log(torch.sqrt(dist2)))

            all_rotation.append(torch.stack([self.gaussians._rotation[-1].clone().detach()] * len(real_nn_idx))) # num_matches x 4

            all_kf_IDs.append(torch.ones(len(real_nn_idx)).long() * current_window[cam_idx]) # num_matches
            all_n_obs.append(torch.zeros(len(real_nn_idx)).long()) # num_matches

        kwargs = {
            "new_xyz": torch.cat(all_xyzs, dim=0),  # (num_matches * nn) x 3
            "new_features_dc": torch.cat(all_feat_dc, dim=0),  # (num_matches * nn) x 3 x 1
            "new_features_rest": torch.cat(all_feat_rest, dim=0),  # (num_matches * nn) x 3 x 15
            "new_opacities": torch.cat(all_opacity, dim=0),  # (num_matches * nn) x 1
            "new_scaling": torch.cat(all_scaling, dim=0),
            "new_rotation": torch.cat(all_rotation, dim=0),
            "new_kf_ids": torch.cat(all_kf_IDs, dim=0),
            "new_n_obs": torch.cat(all_n_obs, dim=0),
        }

        # print("九天大阅兵")
        # print("new_xyz", kwargs["new_xyz"].shape)
        # print("new_features_dc", kwargs["new_features_dc"].shape)
        # print("new_features_rest", kwargs["new_features_rest"].shape)
        # print("new_opacities", kwargs["new_opacities"].shape)
        # print("new_scaling", kwargs["new_scaling"].shape)
        # print("new_rotation", kwargs["new_rotation"].shape)
        # print("new_kf_ids", kwargs["new_kf_ids"].shape)
        # print("new_n_obs", kwargs["new_n_obs"].shape)


        return kwargs

    def request_pose_tracking(self, cur_frame_idx, viewpoint):
        msg = ["pose_tracking", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
        self.requested_pose_tracking = 1

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_keyframeII(self, cur_frame_idx, kwargs, viewpoint, current_window, construct_map_count, removed_idx):
        msg = ["keyframeII", cur_frame_idx, kwargs, viewpoint, current_window, construct_map_count, removed_idx]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        stable_count = 0
        construct_map_count = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                toc.record()
                torch.cuda.synchronize()
                duration = tic.elapsed_time(toc)
                print("Spent time:", duration)

                tic.record()

                # 带我收复山河，再来重奏凯歌
                # if self.requested_pose_tracking == 0: 
                #     self.request_pose_tracking(cur_frame_idx, viewpoint)
                #     continue
                # elif self.requested_pose_tracking == 1:
                #     continue 

                # get_new_instance = self.has_new_instance(
                #     cur_frame_idx, viewpoint
                # )
                get_new_instance = False

                # print(cur_frame_idx , '/', len(self.dataset), ",", self.requested_pose_tracking)
                print(cur_frame_idx , '/', len(self.dataset), ", has new ins?", get_new_instance)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    # self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf

                
                if create_kf | get_new_instance:
                    stable_count = 0
                    construct_map_count += 1
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"][0],
                        opacity=render_pkg["opacity"][0],
                        init=False,
                    )
                    

                    # self.request_keyframe(
                    #     cur_frame_idx, viewpoint, self.current_window, depth_map
                    # )

                    new_kwargs = self.init_with_corr(self.current_window)
                    self.request_keyframeII(
                        cur_frame_idx, new_kwargs, viewpoint, self.current_window, construct_map_count, removed
                    )
                else:
                    stable_count += 1
                    if stable_count > 1:
                        self.cleanup(cur_frame_idx-1)
                cur_frame_idx += 1
                self.requested_pose_tracking = 0

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    # No GUI, error reported if we run the following evaluate!!!!!!!!!!!!

                    # eval_ate(
                    #     self.cameras,
                    #     self.kf_indices,
                    #     self.save_dir,
                    #     cur_frame_idx,
                    #     monocular=self.monocular,
                    # )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "pose_tracking":
                    self.sync_backend(data)
                    self.requested_pose_tracking = 2

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
