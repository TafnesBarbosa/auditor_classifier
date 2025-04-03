import numpy as np
from plyfile import PlyData
from gsplat import rasterization
from torch import tensor, float32, int32, exp, sigmoid, cat, clamp, zeros, bmm, unsqueeze, sqrt, arange, inverse, exp, flip, cumprod, full, where, cumsum, diff, nonzero, triu, ones
from torch import max as max_t
from torch import abs as abs_t
from torch import sum as sum_t
from torch.nn.utils.rnn import pad_sequence
import json
from PIL import Image
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
import open3d as o3d
import argparse
import os

def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    # return sh
    return sh * C0 + 0.5

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    scale_factor = 1 / float(max_t(abs_t(optimized_camera_to_world[:, :3, 3])))
    T = optimized_camera_to_world[:, :3, 3:4] * scale_factor # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -bmm(R_inv, T)
    viewmat = zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def read_ply(ply_file, device):
    ply = PlyData.read(ply_file)
    vertex_data = ply["vertex"]

    means = tensor(np.vstack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).T, device=device)
    scales = tensor(np.vstack((vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"])).T, device=device)
    colors = tensor(np.vstack((
        vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"],
        vertex_data["f_rest_0"], vertex_data["f_rest_1"], vertex_data["f_rest_2"],
        vertex_data["f_rest_3"], vertex_data["f_rest_4"], vertex_data["f_rest_5"],
        vertex_data["f_rest_6"], vertex_data["f_rest_7"], vertex_data["f_rest_8"],
        vertex_data["f_rest_9"], vertex_data["f_rest_10"], vertex_data["f_rest_11"],
        vertex_data["f_rest_12"], vertex_data["f_rest_13"], vertex_data["f_rest_14"],
        vertex_data["f_rest_15"], vertex_data["f_rest_16"], vertex_data["f_rest_17"],
        vertex_data["f_rest_18"], vertex_data["f_rest_19"], vertex_data["f_rest_20"],
        vertex_data["f_rest_21"], vertex_data["f_rest_22"], vertex_data["f_rest_23"],
        vertex_data["f_rest_24"], vertex_data["f_rest_25"], vertex_data["f_rest_26"],
        vertex_data["f_rest_27"], vertex_data["f_rest_28"], vertex_data["f_rest_29"],
        vertex_data["f_rest_30"], vertex_data["f_rest_31"], vertex_data["f_rest_32"],
        vertex_data["f_rest_33"], vertex_data["f_rest_34"], vertex_data["f_rest_35"],
        vertex_data["f_rest_36"], vertex_data["f_rest_37"], vertex_data["f_rest_38"],
        vertex_data["f_rest_39"], vertex_data["f_rest_40"], vertex_data["f_rest_41"],
        vertex_data["f_rest_42"], vertex_data["f_rest_43"], vertex_data["f_rest_44"],
    )).T, device=device).reshape(scales.shape[0], -1, 3)
    opacities = tensor(vertex_data["opacity"], device=device)
    quats = tensor(np.vstack((vertex_data["rot_0"], vertex_data["rot_1"], vertex_data["rot_2"], vertex_data["rot_3"])).T, device=device)
    
    # return means, quats, exp(scales), SH2RGB(colors), sigmoid(opacities)
    return means, quats, exp(scales), colors, sigmoid(opacities)

def rasterize(means, quats, scales, opacities, colors, viewmats, Ks, width, height, device):
    render, alphas, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, tile_size=16, render_mode="RGB+D", sh_degree=3, absgrad=True)

    return render, alphas, meta

def read_info(info_path):
    with open(info_path) as file:
        info = json.load(file)
    return info

def read_camera_data(transforms_file, device, frame):
    transforms = read_info(transforms_file)
    downscale = 1
    fx = transforms["fl_x"] 
    fy = transforms["fl_y"]
    cx = transforms["cx"] // downscale
    cy = transforms["cy"] // downscale
    Ks = tensor(np.array([[
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]]), device=device, dtype=float32)

    width, height = transforms["w"] // downscale, transforms["h"] // downscale

    w2cs = get_poses(np.array(transforms["frames"]), device)
    w2cs, _ = auto_orient_and_center_poses(w2cs.cpu(), 'up', 'poses')
    w2cs = make_w2cs_homogenous(w2cs, device)
    w2cs = get_viewmat(w2cs)
    world2cam = unsqueeze(w2cs[frame], 0)
    return Ks, width, height, world2cam

def get_poses(frames, device):
    poses = []
    for frame in frames:
        poses.append(frame["transform_matrix"])
    return tensor(np.array(poses), device=device, dtype=float32)

def make_w2cs_homogenous(w2cs, device):
    w2cs_new = []
    for i in range(0,w2cs.shape[0]):
        w2cs_new.append(cat([w2cs[i].cpu(), tensor([[0, 0, 0, 1]])], 0))
    return tensor(np.array(w2cs_new), dtype=float32, device=device)

def extract_pixel_from_tile(flattens_all, isect_offsets_diff, opacities_all, means_2d_all, isect_ids, tile_ids, tile_width, tile_size, opacity_threshold, depths_all, conics_all, width, height):
    # depths = (isect_ids & 0xFFFFFFFF).to(int32).view(float32)
    tile_ids_all = (isect_ids >> 32) & 0xFFFF

    # depths_in_tile = depths[tile_ids==tile_id]
    flatten_ids = get_flatten_indices(flattens_all, tile_ids_all, tile_ids)
    depths_in_tile = depths_all[flatten_ids]
    opacities_in_tile = opacities_all[flatten_ids]
    means_2d_in_tile = means_2d_all[flatten_ids]
    conics_in_tile = conics_all[flatten_ids]

    pixel_width, pixel_height = compute_mean_pixel_in_area(tile_ids, tile_width, tile_size, width, height)

    if tile_ids[-1] < len(isect_offsets_diff) - 1:
        isect_offsets_idx = isect_offsets_diff[tile_ids]
    else:
        isect_offsets_idx = cat((isect_offsets_diff[tile_ids[:-1]], tensor([len(tile_ids_all)-isect_offsets_diff.sum().item()], dtype=int32, device=0)), 0)
    alphas_bef = f2d(pixel_width, pixel_height, conics_in_tile, means_2d_in_tile, opacities_in_tile, isect_offsets_idx, depths_in_tile)
    isect_offsets_idx_cum_sum = cumsum(isect_offsets_idx, dim=0)
    alphas = [alphas_bef[:isect_offsets_idx_cum_sum[0]]]
    for i in range(1, len(isect_offsets_idx_cum_sum)):
        alphas.append(alphas_bef[isect_offsets_idx_cum_sum[i-1]:isect_offsets_idx_cum_sum[i]])
    alphas = pad_sequence(alphas, batch_first=True, padding_value=0).to(0)
    transmittance = get_transmittance(alphas)
    # print(transmittance.shape, alphas.shape)
    acc_opacity = triu(alphas.unsqueeze(1) * transmittance)
    acc_opacity = get_cum_sum(acc_opacity)
    mask = acc_opacity < opacity_threshold
    idx = nonzero(mask)
    idx = [idx[idx[:, 0] == i] for i in range(acc_opacity.size(0))]
    # idx = mask.nonzero(as_tuple=True)[0][0] if mask.any() else 0
    # idx = idx - 1 if idx != 0 else idx
    idx = [*map(lambda t: t[0][1].item() if len(t) != 0 else 0, idx)]

    return pixel_width, pixel_height, depths_in_tile[idx]

def get_flatten_indices(flattens_all, tile_ids_all, tile_ids):
    mask = (tile_ids_all == tile_ids[0])
    for i in range(1, len(tile_ids)):
        mask |= (tile_ids_all == tile_ids[i])
    return flattens_all[mask]

def get_transmittance(x):
    x = triu(x.unsqueeze(1).repeat(1,x.shape[1],1))
    x = 1 - x
    # x = flip(cumprod(flip(x, dims=[1]), dim=1), dims=[1])
    # x = cumprod(x, dim=1)
    x = cumprod(x, dim=2)
    # x = x[:, 1:]
    # x = x[:, :-1]
    x = x[:,:, :-1]
    # x = cat((x, tensor([1], dtype=float32, device=0).repeat(x.shape[0],1)), dim=1)
    # x = cat((tensor([1], dtype=float32, device=0).repeat(x.shape[0],1), x), dim=1)
    x = cat((tensor([1], dtype=float32, device=0).repeat(x.shape[1], 1).unsqueeze(0).repeat(x.shape[0],1,1), x), dim=2)
    return x

def get_cum_sum(x):
    # return flip(cumsum(flip(x, dims=[1]), dim=1), dims=[1])
    # return cumsum(x, dim=1)
    return sum_t(x, dim=2)

def f2d(u, v, conics, means, opacities, isect_offsets_idx, depths):
    p1 = (cat((u.unsqueeze(0), v.unsqueeze(0)), 0).T)
    p = p1[0].unsqueeze(0).repeat(isect_offsets_idx[0], 1, 1)
    for i in range(1, len(isect_offsets_idx)):
        p = cat((p, p1[i].unsqueeze(0).repeat(isect_offsets_idx[i], 1, 1)), 0)
    p.to(0)
    # print(p.shape, depths.unsqueeze(1).shape)
    p = p / depths.unsqueeze(1).unsqueeze(1)
    means = means.unsqueeze(1).to(0)
    covar = full((conics.shape[0], 2, 2), 0, dtype=float32, device=0)
    covar[:, 0, 0] = conics[:, 0]
    covar[:, 0, 1] = conics[:, 1]
    covar[:, 1, 0] = conics[:, 1]
    covar[:, 1, 1] = conics[:, 2]
    covar.to(0)
    res = exp(bmm(bmm((p-means), covar), (p-means).transpose(1,2)) * (-1) / 2).squeeze(1,2)
    return res * opacities

def compute_area_proportion(circle_center, circle_radius, tile_id, tile_width, tile_size):
    first_height = tile_id // tile_width
    first_width = tile_id - tile_width * first_height

    first_height *= tile_size
    first_width *= tile_size

    tensor_width = arange(first_width, first_width + tile_size).repeat(tile_size, 1).to(0)
    tensor_height = arange(first_height, first_height + tile_size).repeat(tile_size, 1).T.to(0)

    d = sqrt((tensor_width - circle_center[0])**2 + (tensor_height - circle_center[1])**2)
    return ((d <= circle_radius).sum().item()) / (tile_size ** 2)
    
def compute_mean_pixel_in_area(tile_ids, tile_width, tile_size, width, height):
    first_height = tile_ids // tile_width
    first_width = tile_ids - tile_width * first_height

    first_height *= tile_size
    first_width *= tile_size

    pixels_width = first_width + tile_size // 2
    pixels_height = first_height + tile_size // 2

    mask_width = width - first_width < tile_size
    pixels_width[mask_width] = first_width[mask_width] + (width - first_width[mask_width]) // 2
    mask_height = width - first_width < tile_size
    pixels_height[mask_height] = first_height[mask_height] + (height - first_height[mask_height]) // 2

    return pixels_width, pixels_height

def extract_vertex_from_tile(meta, tile_id, opacity_threshold, area_threshold, Ks, w2c):
    u_in_tile, v_in_tile, z_in_tile = extract_pixel_from_tile(
        flattens_all=meta['flatten_ids'], 
        isect_offsets_diff=diff(meta['isect_offsets'].squeeze(0).reshape(-1)), 
        opacities_all=meta['opacities'], 
        means_2d_all=meta['means2d'], 
        isect_ids=meta['isect_ids'], 
        tile_ids=tile_id, 
        tile_width=meta['tile_width'], 
        tile_size=meta['tile_size'], 
        opacity_threshold=opacity_threshold,
        depths_all=meta['depths'],
        conics_all=meta['conics'],
        width=meta['width'],
        height=meta['height']
    )

    R = w2c[:, :3]
    t = w2c[:, 3].unsqueeze(1)
    xyz = cat((u_in_tile.unsqueeze(0), v_in_tile.unsqueeze(0), z_in_tile.unsqueeze(0)))
    mu = inverse(R) @ (inverse(Ks) @ xyz - t)
    # mu = inverse(R) @ (inverse(Ks) @ tensor(np.array([[u_in_tile, v_in_tile, z_in_tile.item()]]), dtype=float32, device=0).T - t)
    return mu, u_in_tile, v_in_tile

def extract_vertex_from_tile_in_depths(meta, depths, tile_id, Ks, w2c):
    u_in_tile, v_in_tile = compute_mean_pixel_in_area(
        tile_id,
        meta['tile_width'], 
        meta['tile_size'], 
        meta['width'],
        meta['height']
    )
    z_in_tile = []
    for i in range(len(u_in_tile)):
        z_in_tile.append(depths[np.round(v_in_tile[i].cpu().numpy()).astype(int), np.round(u_in_tile[i].cpu().numpy()).astype(int)].astype(np.float64))
    z_in_tile = tensor(np.array(z_in_tile), dtype=float32, device=0)
    # print(z_in_tile.shape)

    R = w2c[:, :3]
    t = w2c[:, 3].unsqueeze(1)
    xyz = cat((u_in_tile.unsqueeze(0), v_in_tile.unsqueeze(0), z_in_tile.unsqueeze(0)))
    mu = inverse(R) @ (inverse(Ks) @ xyz - t)
    # mu = inverse(R) @ (inverse(Ks) @ tensor(np.array([[u_in_tile, v_in_tile, z_in_tile.item()]]), dtype=float32, device=0).T - t)
    return mu, u_in_tile, v_in_tile

def populate_ply_file(pcd, new_points, new_colors):
    existing_points = np.asarray(pcd.points)
    all_points = np.vstack([existing_points, new_points]) if len(existing_points) != 0 else new_points
    pcd.points = o3d.utility.Vector3dVector(all_points)

    existing_colors = np.asarray(pcd.colors)
    all_colors = np.vstack([existing_colors, new_colors]) if len(existing_colors) != 0 else new_colors
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    return pcd

def create_ply_file():
    return o3d.geometry.PointCloud()

def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '█'
    spaces = ' ' * (bar_length - int(progress * bar_length))
    print(f'\rProgress: [{arrow * int(progress * bar_length)}{spaces}] {progress * 100:.2f}%', end='', flush=True)

def get_size(transforms_path):
    info = read_info(transforms_path)
    return len(info['frames'])

def export_splat_file(project_path, project_name):
    splat_ply_path = os.path.join(project_path, 'output_splatfacto')
    if not os.path.exists(os.path.join(splat_ply_path, 'splat.ply')):
        os.system(f"ns-export gaussian-splat --load-config {os.path.join(project_path, 'output_splatfacto', project_name, 'splatfacto', '*', 'config.yml')} --output-dir {splat_ply_path}")

def get_tile_ids(tile_ids_bigger, divide):
    size = len(tile_ids_bigger) // divide
    tile_ids = []
    for i in range(divide):
        tile_ids.append(tile_ids_bigger[i*size:(i+1)*size])
    if size * divide < len(tile_ids_bigger):
        tile_ids.append(tile_ids_bigger[divide*size:])
    return tile_ids

def extract_pcd_from_tiles(meta, opacity_threshold, area_threshold, Ks, world2cam, rgbs, pcd):
    divide_width = 2
    for j in range(meta['tile_height']):
        tile_ids = arange(j * meta['tile_width'], (j+1) * meta['tile_width']).to(0)
        tile_ids = get_tile_ids(tile_ids, divide_width)
        if j % 10 == 0:
            print_progress_bar(j, meta['tile_height'])
        for tile_id in tile_ids:
            mu, u, v = extract_vertex_from_tile(meta, tile_id, opacity_threshold, area_threshold, Ks[0], world2cam[0, :3, :])

            rgbs_ply = []
            for i in range(len(u)):
                rgbs_ply.append(rgbs[np.round(v[i].cpu().numpy()).astype(int), np.round(u[i].cpu().numpy()).astype(int), :].astype(np.float64))
            pcd = populate_ply_file(pcd, mu.cpu().numpy().T, np.array(rgbs_ply))
    return pcd

def extract_pcd_from_rendered_image_depths(meta, rgbs, depths, pcd, Ks, world2cam):
    divide_width = 2
    for j in range(meta['tile_height']):
        tile_ids = arange(j * meta['tile_width'], (j+1) * meta['tile_width']).to(0)
        tile_ids = get_tile_ids(tile_ids, divide_width)
        if j % 10 == 0:
            print_progress_bar(j, meta['tile_height'])
        
        for tile_id in tile_ids:
            mu, u, v = extract_vertex_from_tile_in_depths(meta, depths, tile_id, Ks[0], world2cam[0, :3, :])

            rgbs_ply = []
            for i in range(len(u)):
                rgbs_ply.append(rgbs[np.round(v[i].cpu().numpy()).astype(int), np.round(u[i].cpu().numpy()).astype(int), :].astype(np.float64))
            pcd = populate_ply_file(pcd, mu.cpu().numpy().T, np.array(rgbs_ply))
    return pcd

def main(project_path, project_name, opacity_threshold):
    ply_path = os.path.join(project_path, "output_splatfacto/splat.ply")
    transforms_path = os.path.join(project_path, "transforms.json")
    output_ply_file = os.path.join(project_path, "output_splatfacto/output_ply_file.ply")
    output_ply_file_gaus = os.path.join(project_path, "output_splatfacto/output_gauss_means_ply_file.ply")
    device = 0
    # tile_id = 0
    opacity_threshold = opacity_threshold
    area_threshold = 0.5

    print("Exporting gaussian splat .ply to extract normal .ply if it is not exported yet")
    export_splat_file(project_path, project_name)

    print('Reading gaussian splat .ply')
    means, quats, scales, colors, opacities = read_ply(ply_path, device)

    pcd = create_ply_file()
    num_frames = get_size(transforms_path)
    step_frames = 10
    os.system(f"mkdir {project_path}/rendered_images")
    for frame in range(0, num_frames, step_frames):
        print(f'\nProcessing frame {frame} of {num_frames} to extract .ply file')

        print('Reading cameras')
        Ks, width, height, world2cam = read_camera_data(transforms_path, device, frame)

        print("Rasterizing")
        rgbs, _, meta = rasterize(means, quats, scales, opacities, colors, world2cam, Ks, width, height, device)
        acc_depths = rgbs[:,:,:,3]
        rgbs = rgbs[:,:,:,:3]
        # print(meta)

        
        Image.fromarray((clamp(rgbs, 0, 1).cpu().numpy()[0] * 255).astype(np.uint8)).save(f'{project_path}/rendered_images/image_{frame}.png')

        rgbs = clamp(rgbs, 0, 1)
        rgbs = rgbs.cpu().numpy()[0]
        # rgbs_255 = rgbs * 255
        # rgbs_255 = rgbs_255.astype(np.uint8)
        # image = Image.fromarray(rgbs_255)
        # image.save(f'{project_path}/rendered_images/image_{frame}.png')

        acc_depths = acc_depths.cpu().numpy()[0]

        print('Extracting point cloud\n')
        # pcd = extract_pcd_from_tiles(meta, opacity_threshold, area_threshold, Ks, world2cam, rgbs, pcd)
        pcd = extract_pcd_from_rendered_image_depths(meta, rgbs, acc_depths, pcd, Ks, world2cam)
    o3d.io.write_point_cloud(output_ply_file, pcd)

    pcd = create_ply_file()
    pcd = populate_ply_file(pcd, means.cpu().numpy(), colors.cpu().numpy())
    o3d.io.write_point_cloud(output_ply_file_gaus, pcd)

parser = argparse.ArgumentParser(description="Script with argparse options")
# Add arguments
parser.add_argument("-pp", "--project_path", type=str, help="Folder with project folder: colmap, images and outputs. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-pn", "--project_name", type=str, help="Name of project folder", default=None)
parser.add_argument("-ot", "--opacity_threshold", type=float, help="Accumulative opacity threshold to define where to get the depth of image color", default=0.75)
# Parse arguments
args = parser.parse_args()

main(args.project_path, args.project_name, args.opacity_threshold)