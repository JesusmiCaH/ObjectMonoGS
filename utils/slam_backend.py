import random
import time
import numpy as np 

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_tracking, get_loss_mapping_combined
from romatch import roma_outdoor, roma_indoor

import os
from PIL import Image


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # Load matcher
        self.roma_model = roma_indoor(device=self.device)
        self.roma_model.upsample_preds = False
        self.roma_model.symmetric = False

        self.removed_frames = [0]

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )

        # self.rigid_tracking_itr_num = self.config["Training"]["rigid_tracking_itr_num"]

        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )


    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss_init = get_loss_mapping_combined(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                
                loss_mapping += get_loss_mapping_combined(
                    self.config, image, depth, viewpoint, opacity
                )
                # print("loss is", loss_mapping.item())
                # print("depth对比", depth[0][0,222:224, 222:224], viewpoint.depth[222:224, 222:224])
                # print("rgb对比", image[0][:, 222:224,222:224], viewpoint.original_image[:, 222:224,222:224])

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                # n_touched is one-time use only

            # Doing the same thing compared with the above part
            # But randomly select 2 previous viewpoints to optimize (not have to be kf)
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping_combined(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            # isotropic is to make ellipsoids more spherical
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )

                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.prune(
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

        return gaussian_split


    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"][0],
                render_pkg["visibility_filter"][0],
                render_pkg["radii"][0],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")
    
    def take_a_look(self, gaussians, viewpoint):
        pass

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            # print("nothing to do")
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                # self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                print(data[0])
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "pose_tracking":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]

                    self.gaussians.set_frame(cur_frame_idx)
                    self.gaussians.densification_newframe(viewpoint)
                    keep_tracking = (
                        (self.gaussians.get_ins_ids>0).any()        
                    )
                    # print("让我看看", (self.gaussians.get_ins_ids>0).unique())
                    if not keep_tracking:
                        print(cur_frame_idx, "No instance to track") 
                    else:
                        print(cur_frame_idx, "Tracking instances", end=" ")
                        print(np.unique(viewpoint.segment_map))

                        for tracking_itr in range(self.rigid_tracking_itr_num):
                            render_pkg = render(
                                viewpoint, self.gaussians, self.pipeline_params, self.background
                            )
                            image, depth, opacity = (
                                render_pkg["render"],
                                render_pkg["depth"],
                                render_pkg["opacity"],
                            )
                            loss_tracking = get_loss_tracking(
                                self.config, image[2], depth[2], opacity[2], viewpoint, focus_part = "rigid"
                            )
                            loss_tracking.backward()
                            
                            prev_ins_means = self.gaussians.get_ins_means[cur_frame_idx]
                            prev_ins_quats = self.gaussians.get_ins_quats[cur_frame_idx]
                            self.gaussians.rigid_optimizer.step()
                            converged = (
                                torch.norm(self.gaussians.get_ins_means[cur_frame_idx] - prev_ins_means) < 1e-4 and
                                torch.norm(self.gaussians.get_ins_quats[cur_frame_idx] - prev_ins_quats) < 1e-4
                            )
                            
                            if converged:
                                break
                    self.push_to_frontend("pose_tracking")                            

                    
                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    print(current_window)
                    depth_map = data[4]     # depth information is preprocessed in the frontend

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)

                    # Save the rendered image as an image file in the "./img_result" directory
                    output_dir = "./img_result"
                    os.makedirs(output_dir, exist_ok=True)
                    # Render the image using the current viewpoint
                    
                    render_pkg = render(
                        data[2], self.gaussians, self.pipeline_params, self.background
                    )
                    image = render_pkg["render"][0]
                    # Convert the rendered image tensor to a PIL image and save it
                    rendered_image = image.detach().cpu().numpy().transpose(1, 2, 0)  # Assuming CHW format
                    rendered_image = (rendered_image * 255).clip(0, 255).astype("uint8")  # Scale to 0-255
                    
                    gt_image = data[2].original_image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image = (gt_image * 255).clip(0, 255).astype("uint8")
                    # Save the rendered image
                    output_path = os.path.join(output_dir, f"rendered_image_{cur_frame_idx}.png")
                    Image.fromarray(rendered_image).save(output_path)
                    # Save the ground truth image
                    gt_output_path = os.path.join(output_dir, f"gt_image_{cur_frame_idx}.png")
                    Image.fromarray(gt_image).save(gt_output_path)
                    
                    
                    self.push_to_frontend("keyframe")
                
                elif data[0] == "keyframeII":
                    cur_frame_idx = data[1]
                    kwargs = data[2]
                    viewpoint = data[3]
                    current_window = data[4]
                    construct_map_count = data[5]
                    removed_idx = data[6]
                    # print("先爽爽",self.removed_frames)
                    # print("先试试",removed_idx, torch.isin(self.gaussians.unique_kfIDs, torch.tensor(self.removed_frames)))
                    
                    print("You know what you are??????????",self.gaussians.unique_kfIDs)
                    
                    if removed_idx is not None:
                        self.removed_frames.append(removed_idx)
                    # if construct_map_count % 8 == 0:
                    #     print("重生！！！！！！！！！！！！！")
                    self.gaussians.prune_points(torch.isin(self.gaussians.unique_kfIDs, torch.tensor(self.removed_frames)))
                    
                    self.gaussians.densification_postfix(**kwargs)
                    self.current_window = current_window
                    self.viewpoints[cur_frame_idx] = viewpoint

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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

                    # if (construct_map_count >= 8) & (construct_map_count%8 == 0):
                    #     print("开始优化了", construct_map_count, (construct_map_count >= 8), (construct_map_count%8 == 0))
                    #     self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    #     iter_per_kf = 10
                    #     self.map(self.current_window, iters=iter_per_kf)
                    #     self.map(self.current_window, prune=True)

                    # Save the rendered image as an image file in the "./img_result" directory
                    output_dir = "./img_result"
                    os.makedirs(output_dir, exist_ok=True)
                    # Render the image using the current viewpoint
                    
                    for k_idx in self.current_window:
                        render_pkg = render(
                            self.viewpoints[k_idx], self.gaussians, self.pipeline_params, self.background
                        )
                        self.occ_aware_visibility[k_idx] = render_pkg["n_touched"]

                        image = render_pkg["render"][0]
                        # Convert the rendered image tensor to a PIL image and save it
                        rendered_image = image.detach().cpu().numpy().transpose(1, 2, 0)  # Assuming CHW format
                        rendered_image = (rendered_image * 255).clip(0, 255).astype("uint8")  # Scale to 0-255
                        # Save the rendered image
                        output_path = os.path.join(output_dir, f"rendered_image_{cur_frame_idx}_view{k_idx}.png")
                        Image.fromarray(rendered_image).save(output_path)
                        # Save the rendered depth
                        depth = render_pkg["depth"][0].detach().squeeze().cpu().numpy()
                        print("depth 是", depth[222:224, 222:224])
                        depth = (1.0 / depth * 255).clip(0, 255).astype("uint8")
                        depth_output_path = os.path.join(output_dir, f"depth_image_{cur_frame_idx}_view{k_idx}.png")
                        Image.fromarray(depth).save(depth_output_path)
                    

                    gt_image = self.viewpoints[cur_frame_idx].original_image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image = (gt_image * 255).clip(0, 255).astype("uint8")
                    # Save the ground truth image
                    gt_output_path = os.path.join(output_dir, f"gt_image_{cur_frame_idx}.png")
                    Image.fromarray(gt_image).save(gt_output_path)
                    
                    gt_depth = self.viewpoints[cur_frame_idx].depth
                    gt_depth[gt_depth == 0] = 1e6
                    gt_depth = (1.0 / gt_depth * 255).clip(0, 255).astype("uint8")
                    # Save the ground truth depth
                    gt_depth_output_path = os.path.join(output_dir, f"gt_depth_image_{cur_frame_idx}.png")
                    Image.fromarray(gt_depth).save(gt_depth_output_path)
                    self.push_to_frontend("keyframe")
 
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
