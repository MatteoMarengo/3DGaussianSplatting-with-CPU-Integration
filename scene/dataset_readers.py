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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3Dv4.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 1_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##############################################################################################################
##############################################################################################################

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import namedtuple

# def focal2fov(focal_length, size):
#     return 2 * np.arctan(size / (2 * focal_length)) * (180 / np.pi)

# def fov2focal(fov, size):
#     return size / (2 * np.tan(np.deg2rad(fov) / 2))

def readCamerasFromTransformsMODI(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            if fovx:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            else:
                fovy = focal2fov(f_y, image.size[1])
                fovx = focal2fov(f_x, image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,image = image,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3dv2.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfSyntheticInfoMODI(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODI(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODI(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "lidar_pcd.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 100_000
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##############################################################################################################
# MODI2
##############################################################################################################

def readCamerasFromTransformsMODI2(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            size0 = 1600
            size1 = 900

            ########################################

            # if fovx:
            #     fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            # else:
            #     fovy = focal2fov(f_y, image.size[1])
            #     fovx = focal2fov(f_x, image.size[0])

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODI2(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODI2(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODI2(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_500.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/GAUSSIAN-SPLATTING-VERSIONS/3DGS-CPU-ADAPTED/data-3DGS/NuScenes-SMALL-SCENE/inliers_plane_1_0.5.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_200.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/GAUSSIAN-SPLATTING-VERSIONS/3DGS-CPU-ADAPTED/data-3DGS/points3d_all_frames_centered_3_200.ply")
    ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/1_DATASET_LOADING/001/combined_point_cloud.ply"
    ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/5_TRANSFORM_INTO_NERFS/downsampled_file_001_0.5.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    #ply_path = "rannanan"
    # ply_path = "random_PANDASET.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        #num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


##############################################################################################################
# MODI3

def readCamerasFromTransformsMODI3(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])

            # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            c2w = c2w.copy()
            adapted_c2w = np.zeros_like(c2w)
            # Adjust axes
            adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(adapted_c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos


def readNerfSyntheticInfoMODI3(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODI3(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODI3(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_500.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/GAUSSIAN-SPLATTING-VERSIONS/3DGS-CPU-ADAPTED/data-3DGS/NuScenes-SMALL-SCENE/inliers_plane_1_0.5.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_200.ply")
    ply_path = os.path.join(path, "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/GAUSSIAN-SPLATTING-VERSIONS/3DGS-CPU-ADAPTED/data-3DGS/points3d_all_frames_centered_3_200.ply")
    ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/1_DATASET_LOADING/001/combined_point_cloud.ply"
    ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/5_TRANSFORM_INTO_NERFS/downsampled_file_001_0.5.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    #ply_path = "rannanan"
    # ply_path = "random_PANDASET.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        #num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################

def readCamerasFromTransformsMODITEST1(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos


def readNerfSyntheticInfoMODITEST1(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST1(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST1(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        #num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


############################################################################################################################
############################################################################################################################
### TEST 2 ###
### Change current_from_car0 
### Keep OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST2(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]
            R = np.transpose(c2w[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = c2w[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST2(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST2(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST2(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 3 ###
### Keep current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST3(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])
            # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            c2w = c2w.copy()
            adapted_c2w = np.zeros_like(c2w)
            # Adjust axes
            adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(adapted_c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST3(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST3(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST3(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 4 ###
### Change current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST4(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            c2w = np.array(frame["transform_matrix"])
            # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            c2w = c2w.copy()
            adapted_c2w = np.zeros_like(c2w)
            # Adjust axes
            adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            R = np.transpose(adapted_c2w[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = adapted_c2w[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos    

def readNerfSyntheticInfoMODITEST4(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST4(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST4(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 5 ###
### Change current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST5(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            c2w = np.array(frame["transform_matrix"])
            c2w = np.linalg.inv(c2w)
            # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            c2w = c2w.copy()
            adapted_c2w = np.zeros_like(c2w)
            # Adjust axes
            adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST5(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST5(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST5(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 6 ###
### Change current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST6(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            c2w = np.array(frame["transform_matrix"])
            c2w = np.linalg.inv(c2w)
            # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST6(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST6(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST6(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 6 ###
### Change current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST7(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            w2c = np.array(frame["transform_matrix"])
            # c2w = np.linalg.inv(c2w)
            # # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            if fovx:
                fovy = focal2fov(fov2focal(fovx, size0), size1)
            else:
                fovy = focal2fov(f_y, size1)
                fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST7(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST7(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST7(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

############################################################################################################################
############################################################################################################################
### TEST 8 ###
### Change current_from_car0 
### Change OpenGL -> COLMAP conversion

def readCamerasFromTransformsMODITEST8(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents.get("camera_angle_x", None)  # Get FOV from the JSON if available
        fovx = None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            w2c = np.array(frame["transform_matrix"])
            # c2w = np.linalg.inv(c2w)
            # # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            # cx = intrinsics[0, 2]
            # cy = intrinsics[1, 2]

            ####### LOAD THE IMAGE PART #######
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            size0 = 1600
            size1 = 900

            ########################################

            # if fovx:
            #     fovy = focal2fov(fov2focal(fovx, size0), size1)
            # else:
            fovy = focal2fov(f_y, size1)
            fovx = focal2fov(f_x, size0)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoMODITEST8(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsMODITEST8(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsMODITEST8(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    # ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE0-ALL-FRAMES/points3d_all_frames_centered_0_255.ply"
    # ply_path = "random_initinitnsscene0.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 600_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        xyz = np.random.random((num_pts, 3)) * 200
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


#################################################################################################################
#### TEST 1 ####
## My CAMERAS World To Camera
## ORIGINAL SCRIPT (we perform the original script)

def readCamerasFromTransformsCOMPTEST1(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            c2w = np.array(frame["transform_matrix"])
            # c2w = np.linalg.inv(c2w)
            # # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]



            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            size0 = 1600
            size1 = 900

            fovy = focal2fov(f_y, size1)
            fovx = focal2fov(f_x, size0)


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoCOMPTEST1(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsCOMPTEST1(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsCOMPTEST1(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    # ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

#################################################################################################################
#### TEST 2 ####
## My CAMERAS World To Camera
## ORIGINAL SCRIPT but now we consider that load cameras are W2C and not C2W

def readCamerasFromTransformsCOMPTEST2(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            w2c = np.array(frame["transform_matrix"])
            c2w = np.linalg.inv(w2c)
            # c2w = np.linalg.inv(c2w)
            # # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]



            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            size0 = 1600
            size1 = 900

            fovy = focal2fov(f_y, size1)
            fovx = focal2fov(f_x, size0)


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoCOMPTEST2(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsCOMPTEST2(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsCOMPTEST2(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    # ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 200_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

#################################################################################################################
#### TEST 3 ####
## My CAMERAS World To Camera
## ORIGINAL SCRIPT but now we consider that load cameras are W2C and not C2W
## We now keep the original transformation 


def readCamerasFromTransformsCOMPTEST3(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = os.path.join(path, frame["file_path"])

            # Read the transform matrix
            # IT IS WORLD TO CAMERA
            w2c = np.array(frame["transform_matrix"])
            c2w = np.linalg.inv(w2c)
            # c2w = np.linalg.inv(c2w)
            # # # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            # # Adapt from X forward, Z up (your coordinate system) to COLMAP (X right, Y down, Z forward)
            # c2w = c2w.copy()
            # adapted_c2w = np.zeros_like(c2w)
            # # Adjust axes
            # adapted_c2w[:, 0] = -c2w[:, 1]  # Your Y-axis (left) becomes COLMAP X-axis (right)
            # adapted_c2w[:, 1] = -c2w[:, 2]  # Your Z-axis (up) becomes COLMAP Y-axis (down)
            # adapted_c2w[:, 2] = c2w[:, 0]   # Your X-axis (forward) becomes COLMAP Z-axis (forward)
            # adapted_c2w[:, 3] = c2w[:, 3]   # Translation remains the same

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read the intrinsics
            intrinsics = np.array(frame["intrinsics"])
            f_x = intrinsics[0, 0]
            f_y = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]



            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            size0 = 1600
            size1 = 900

            fovy = focal2fov(f_y, size1)
            fovx = focal2fov(f_x, size0)


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=image_name, width=size0, height=size1))

    return cam_infos

def readNerfSyntheticInfoCOMPTEST3(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransformsCOMPTEST3(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsCOMPTEST3(path, "transforms_test.json", white_background, extension)
    # print("Reading Transforms")
    # cam_infos = readCamerasFromTransformsMODI(path, "transforms.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "lidar_pcd.ply")
    # ply_path = "/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/PANDASET-TESTS/OFFLINE/downsampled_file_001_0.522.ply"
    # ply_path ='/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/CROP_POINT_CLOUD/output_cropped_x_70_110.ply'
    # ply_path = '/home/mmarengo/workspace/3DGS-URBAN-SCENES-INTERNSHIP/1_NUSCENES-PCD-EXTRACTION/EXTRACTED_PCD/CENTERED/points3d_all_frames_centered_3_250.ply'
    # ply_path = '/home/mmarengo/workspace/SYNTHETIC-NERF-PANDASET/SYNTHETIC-NERF-PANDASET-SCENE001-186/point_cloud_001_186.ply'
    #ply_path = "randomahah.ply"
    # ply_path = "/home/mmarengo/workspace/SYNTHETIC-NERF-NUSCENES/SYNTHETIC-NERF-NUSCENES-SCENE3-ALL-FRAMES/points3d_all_frames_centered_colored_3_250.ply"
    ply_path = "random_600k.ply"
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 600_000
        # num_pts = 10_000
        # The script generates random 3D points inside a cube with bounds [-1.3,1.3] in each dimension

        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

#################################################################################################################
#################################################################################################################

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfoCOMPTEST3
}