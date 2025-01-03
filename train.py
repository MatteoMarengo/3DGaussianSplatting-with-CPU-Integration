# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import ipdb
import numpy as np
from utils.general_utils import PILtoTorch

from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

################################################################################################################################
################################################################################################################################
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print("LOAD THE GAUSSIANS ...")
    gaussians = GaussianModel(dataset.sh_degree)
    print("LOAD THE SCENE ...")
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # bg_color is now yellow
    # bg_color is now orange
    # bg_color = [1, 0.5, 0]
     
    # bg_color = [1, 0, 0]
    # bg_color = [1, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # print("First Iteration: ", first_iter)
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

        # Pick a random Camera
        # print("Pick a random Camera")
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # print("Before RENDER")
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
 
        # print("image shape: ", image.shape)
        # image = image.permute(1, 2, 0)
        # print("image shape: ", image.shape)
        # plt.imsave("image_orange_reindeer.png", image.detach().cpu().numpy())

        # print("Before GT IMAGE")
        # Loss
        ####### LOAD THE IMAGE PART #######
        gimage = Image.open(viewpoint_cam.image_path)

        im_data = np.array(gimage.convert("RGBA"))

        bga = np.array([1, 1, 1]) # if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bga * (1 - norm_data[:, :, 3:4])
        gimage = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

        ########################################
        resized_image_rgb = PILtoTorch(gimage, viewpoint_cam.resolution)
        gimage = resized_image_rgb[:3, ...]
        loaded_mask = None

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

        original_image = gimage.clamp(0.0, 1.0).to("cuda")
        image_width = original_image.shape[2]
        image_height = original_image.shape[1]

        if loaded_mask is not None:
            original_image *= loaded_mask.to("cuda")
        else:
            original_image *= torch.ones((1, image_height, image_width), device="cuda")

        gt_image = original_image

        # viewpoint_cam.original_image = PILtoTorch(Image.open(dataset.source_path + "/images/" + viewpoint_cam.image_name), (viewpoint_cam.image_width, viewpoint_cam.image_height))
        # gt_image = viewpoint_cam.original_image.cuda()
        # gt_image = viewpoint_cam.original_image.cuda()
        # gt_image = gt_image.permute(1, 2, 0)
        # plt.imsave("gt_image.png", gt_image.detach().cpu().numpy())

        # print(f"gt_image is on cuda: {gt_image.is_cuda}")
        # put image on cuda
        # image = image.to("cuda")
        # print(f"image is on cuda: {image.is_cuda}")

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Store information about the gradients
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

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

       

    gaussians_count = scene.gaussians.get_xyz.shape[0]
    with open(scene.model_path + '/gaussiancount.txt', 'w') as f:
        f.write(str(gaussians_count))

#####################################################################################################################################
#####################################################################################################################################

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
        print("Tensorboard has been found")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

#####################################################################################################################################
#####################################################################################################################################

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('nb_gaussians', scene.gaussians.get_xyz.shape[0],iteration)

        # if(iteration%50==0):
        #     grads_xyz=scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
        #     grads_xyz[grads_xyz.isnan()]=0.0

        #     grads_featuresdc=scene.gaussians.features_dc_gradient_accum / scene.gaussians.denom
        #     grads_featuresdc[grads_featuresdc.isnan()]=0.0

        #     grads_featuresrest=scene.gaussians.features_rest_gradient_accum / scene.gaussians.denom
        #     grads_featuresrest[grads_featuresrest.isnan()]=0.0

        #     grads_scaling=scene.gaussians.scaling_gradient_accum / scene.gaussians.denom
        #     grads_scaling[grads_scaling.isnan()]=0.0

        #     grads_rotation=scene.gaussians.rotation_gradient_accum / scene.gaussians.denom
        #     grads_rotation[grads_rotation.isnan()]=0.0

        #     grads_opacity=scene.gaussians.opacity_gradient_accum / scene.gaussians.denom
        #     grads_opacity[grads_opacity.isnan()]=0.0

        #     tb_writer.add_scalar('xyz_gradmean', grads_xyz.mean(),iteration)
        #     tb_writer.add_scalar('featuresdc_gradmean', grads_featuresdc.mean(),iteration)
        #     tb_writer.add_scalar('featuresrest_gradmean', grads_featuresrest.mean(),iteration)
        #     tb_writer.add_scalar('scaling_gradmean', grads_scaling.mean(),iteration)
        #     tb_writer.add_scalar('rotation_gradmean', grads_rotation.mean(),iteration)
        #     tb_writer.add_scalar('opacity_gradmean', grads_opacity.mean(),iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    #######################################
                    gimage = Image.open(viewpoint.image_path)

                    im_data = np.array(gimage.convert("RGBA"))

                    bg = np.array([1, 1, 1]) # if white_background else np.array([0, 0, 0])

                    norm_data = im_data / 255.0
                    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    gimage = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

                    ########################################
                    resized_image_rgb = PILtoTorch(gimage, viewpoint.resolution)
                    gimage = resized_image_rgb[:3, ...]
                    loaded_mask = None

                    if resized_image_rgb.shape[1] == 4:
                        loaded_mask = resized_image_rgb[3:4, ...]

                    original_image = gimage.clamp(0.0, 1.0).to("cuda")
                    image_width = original_image.shape[2]
                    image_height = original_image.shape[1]

                    if loaded_mask is not None:
                        original_image *= loaded_mask.to("cuda")
                    else:
                        original_image *= torch.ones((1, image_height, image_width), device="cuda")

                    gt_image = original_image
                    #######################################
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])     
                ssim_test /= len(config['cameras'])  

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000,7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000,7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000,30_000, 50_000, 70_000,90_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
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
