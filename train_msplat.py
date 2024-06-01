#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn.functional as F
import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_ms
import sys

sys.path.append('/root/autodl-tmp/RAFT')
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import msplat
import cv2
from utils.sh_utils import eval_sh
from flow_vis import flow_to_image
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_cam = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    
    
    
    ###### RAFT part ######
    padder = None
    RAFT_args = Namespace()
    RAFT_args.model = "/root/autodl-tmp/RAFT/models/raft-things.pth"
    RAFT_args.small = False
    RAFT_args.mixed_precision = True
    RAFT_args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(RAFT_args))
    model.load_state_dict(torch.load(RAFT_args.model))

    model = model.module
    model.cuda()
    model.eval()
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera or choose it as the next of last step
        if viewpoint_cam is None:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = next_viewpoint_cam
        
        # Pick another Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        next_viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        ############# shape adjustment ##############
        if not hasattr(next_viewpoint_cam, "flow_up") or not hasattr(viewpoint_cam, "flow_up"):
            with torch.no_grad():
                if padder is None:
                    padder = InputPadder(next_viewpoint_cam.original_image.shape)
                if viewpoint_cam.image_width % 8 != 0:
                    viewpoint_cam.original_image = padder.pad(viewpoint_cam.original_image)[0]
                if next_viewpoint_cam.image_width % 8 != 0:
                    next_viewpoint_cam.original_image = padder.pad(next_viewpoint_cam.original_image)[0]
                viewpoint_cam.image_width = viewpoint_cam.original_image.shape[2]
                viewpoint_cam.image_height = viewpoint_cam.original_image.shape[1]
                next_viewpoint_cam.image_width = next_viewpoint_cam.original_image.shape[2]
                next_viewpoint_cam.image_height = next_viewpoint_cam.original_image.shape[1]

                flow_low, flow_up = model(viewpoint_cam.original_image.unsqueeze(0)*255, next_viewpoint_cam.original_image.unsqueeze(0)*255, iters=20, test_mode=True)
                flow_up = -flow_up.squeeze(0)
                # flow_up = (flow_up - torch.min(flow_up)) / ( torch.max(flow_up) -  torch.min(flow_up))
                viewpoint_cam.flow_up = flow_up.detach()
            filename = viewpoint_cam.image_name + "_pred.npy"
            gt_path = "/root/autodl-tmp/GeoWizard/geowizard/ffll_output"
            depth_gt = np.load(os.path.join(gt_path, "depth_npy", filename))
            depth_gt = torch.tensor(depth_gt).cuda()
            depth_gt = padder.pad(depth_gt.unsqueeze(0))[0].squeeze(0)
            depth_gt = ((depth_gt - torch.min(depth_gt)) / ( torch.max(depth_gt) -  torch.min(depth_gt)))
            # cv2.imwrite("comp/AAA_depth_gt_{}.jpg".format(viewpoint_cam.image_name), (depth_gt*255).cpu().numpy())
            
            normal_gt = np.load(os.path.join(gt_path, "normal_npy", filename))
            normal_gt = torch.tensor(normal_gt).cuda()
            normal_gt = F.normalize(normal_gt, dim=-1).permute(2,0,1)
            normal_gt = padder.pad(normal_gt)[0].permute(1,2,0)
            viewpoint_cam.depth_gt = depth_gt
            viewpoint_cam.normal_gt = normal_gt
            
            # exit()
        else:
            flow_up = viewpoint_cam.flow_up
            depth_gt = viewpoint_cam.depth_gt
            normal_gt = viewpoint_cam.normal_gt
            
            
        render_ms_pkg = render_ms(viewpoint_cam, next_viewpoint_cam, gaussians, pipe, bg)
        image_msplat, viewspace_point_tensor, visibility_filter, radii = render_ms_pkg["render"], render_ms_pkg["viewspace_points"], render_ms_pkg["visibility_filter"], render_ms_pkg["radii"]
        
            
        image = image_msplat
        # exit()
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        ori_loss = loss
        
        ########### optical flow loss ##########
        # opticalflow_norm = (render_ms_pkg['opticalflow'] - torch.min(render_ms_pkg['opticalflow'])) / ( torch.max(render_ms_pkg['opticalflow']) -  torch.min(render_ms_pkg['opticalflow']))
        # flow_up = (flow_up - torch.min(flow_up)) / ( torch.max(flow_up) -  torch.min(flow_up))
        flow_loss = l1_loss(render_ms_pkg['opticalflow'],  flow_up)
        loss += flow_loss / flow_loss.item() * ori_loss.item() / 2
        
        
        
        ########### depth loss ##########
        image_depth =  render_ms_pkg['depth']
        depth_map = image_depth.squeeze(0)
        normalized_depth_map = ((depth_map - torch.min(depth_map)) / ( torch.max(depth_map) -  torch.min(depth_map)))
        depth_loss = l1_loss(normalized_depth_map, viewpoint_cam.depth_gt)
        # loss += depth_loss / depth_loss.detach().item() * ori_loss.item() 
        
        
        ########### normal loss ##########
        image_normal = render_ms_pkg['normal']
        image_normal = image_normal.permute(1,2,0)
        normal_loss = l1_loss(image_normal, viewpoint_cam.normal_gt)
        # loss += normal_loss / normal_loss.detach().item() * ori_loss.item() / 4
        if iteration % 100 == 0:
            print("loss:", loss.item())
            print("flow loss:", flow_loss.item())
            print("depth_loss:", depth_loss.item())
            print("normal loss:", normal_loss.item())
        loss.backward()

        iter_end.record()

        if iteration % 1000 == 0:
            image_msplat = cv2.cvtColor(image_msplat.permute(1,2,0).cpu().detach().numpy()*255, cv2.COLOR_BGR2RGB, )
            cv2.imwrite("comp/msplat_{}.jpg".format(iteration), image_msplat)
            
            ############## depth visulization #################
            # image_depth =  render_ms_pkg['depth']
            # depth_map = image_depth.squeeze(0)
            # normalized_depth_map = ((depth_map - torch.min(depth_map)) / ( torch.max(depth_map) -  torch.min(depth_map))).cpu().detach().numpy()
            # # # 映射到0-255并转换为uint8
            depth_map_0_255 = (normalized_depth_map * 255).cpu().detach().numpy()
            cv2.imwrite("comp/depth_{}.jpg".format(iteration), depth_map_0_255)
            depth_gt_255 = (viewpoint_cam.depth_gt * 255).cpu().detach().numpy()
            cv2.imwrite("comp/depth_gt_{}.jpg".format(iteration), depth_gt_255)

            ############## normal visulization #################
            
            image_normal = render_ms_pkg['normal']
            image_normal = image_normal.cpu().detach().numpy()
            # 转置tensor以匹配OpenCV的图像格式[H, W, 3]
            normal_map = np.transpose(image_normal, (1, 2, 0))

            normal_map_visualized = ((normal_map + 1) / 2.0 * 255)
            print(normal_map_visualized.shape)
            cv2.imwrite("comp/normal_{}.jpg".format(iteration), normal_map_visualized)
            normal_gt = normal_gt.cpu().detach().numpy()
            normal_gt_visulized = ((normal_gt + 1) / 2.0 * 255)
            cv2.imwrite("comp/normal_gt_{}.jpg".format(iteration), normal_gt_visulized)
            ############## sceneflow visulization #################
            # image_flow = render_ms_pkg['sceneflow']
            # image_flow = ((image_flow - torch.min(image_flow)) / ( torch.max(image_flow) -  torch.min(image_flow))).cpu().detach().numpy()
            # # 转置tensor以匹配OpenCV的图像格式[H, W, 3]
            # flow_map = np.transpose(image_flow, (1, 2, 0))

            # flow_map_visualized = (flow_map * 255).astype(np.uint8)
            # cv2.imwrite("comp/sceneflow_{}.jpg".format(iteration), flow_map_visualized)
            
            ############## opticalflow visulization #################
            image_flow = render_ms_pkg['opticalflow'].permute(1,2,0)
            image_flow = ((image_flow - torch.min(image_flow)) / ( torch.max(image_flow) -  torch.min(image_flow))).cpu().detach().numpy()
            print(image_flow.shape)
            image_flow = flow_to_image(image_flow)
            cv2.imwrite("comp/opticalflow_{}.jpg".format(iteration), image_flow)
            image_flow = flow_up.permute(1,2,0)
            image_flow = ((image_flow - torch.min(image_flow)) / ( torch.max(image_flow) -  torch.min(image_flow))).cpu().detach().numpy()
            image_flow = flow_to_image(image_flow)
            cv2.imwrite("comp/RAFT_opticalflow_{}.jpg".format(iteration), image_flow)
            ############## end visulization #################


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            loss = {"Ll1": Ll1, "total_loss": loss, "depth_loss": depth_loss, "normal_loss": normal_loss, "flow_loss": flow_loss}
            # Log and save
            training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_ms, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for key in loss.keys():
            tb_writer.add_scalar('train_loss_patches/'+key, loss[key].item(), iteration)

        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    next_viewpoint = config['cameras'][(idx+1)%(len(config['cameras']))]
                    ms_pkg = renderFunc(viewpoint, next_viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(ms_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        ### add rendered depth map
                        depth_map =  ms_pkg['depth']
                        normalized_depth_map = ((depth_map - torch.min(depth_map)) / ( torch.max(depth_map) -  torch.min(depth_map)))
                        tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(viewpoint.image_name), normalized_depth_map[None], global_step=iteration)
                        ### add rendered normal map
                        image_normal = ms_pkg['normal']
                        tb_writer.add_images(config['name'] + "_view_{}/render_normal".format(viewpoint.image_name), image_normal[None], global_step=iteration)

                        ## TODO: add optical flow visulization of test data 

                        
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            
                            ### ground thuth of depth map
                            filename = viewpoint.image_name + "_pred.npy"
                            gt_path = "/root/autodl-tmp/GeoWizard/geowizard/ffll_output"
                            depth_gt = torch.tensor(np.load(os.path.join(gt_path, "depth_npy", filename))).unsqueeze(0)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth".format(viewpoint.image_name), depth_gt[None], global_step=iteration)
                            normal_gt = (torch.tensor(np.load(os.path.join(gt_path, "normal_npy", filename))).permute(2,0,1)+1)/2
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_normal".format(viewpoint.image_name), normal_gt[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.iterations = 10000
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
