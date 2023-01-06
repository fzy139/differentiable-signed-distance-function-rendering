import argparse
import numpy as np
import mcubes
import trimesh
import numpy as np
import mitsuba as mi
import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
from pathlib import Path
from pytorch3d.loss import chamfer_distance
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

def read_img(fn, exposure=0, tonemap=True, background_color=None,
             handle_inexistant_file=False):
    if handle_inexistant_file and not os.path.isfile(fn):
        return np.ones((256, 256, 3)) * 0.3
    bmp = mi.Bitmap(fn)
    if tonemap:
        if background_color is not None:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, False))
            background_color = np.array(background_color).ravel()[None, None, :]
            img = img[:, :, :3] + (1.0 - img[..., -1][..., None]) * background_color
        else:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False))
        img = img * 2 ** exposure

        return np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)), 0, 1)
    else:
        return np.array(bmp)

def extract_mesh(src_path):
    loaded_data = np.array(mi.VolumeGrid(src_path))
    print("Volume Size: ", loaded_data.shape)
    vertices, triangles = mcubes.marching_cubes(loaded_data, 0.0)
    vertices /= np.amax(np.amax(vertices, axis=0) - np.amin(vertices, axis=0))
    vertices -= 0.5
    Ry90 = np.array([[0., 0., -1.],
                [0., 1., 0.],
                [1., 0., 0.]]) 
    vertices = (Ry90 @ vertices[...,None]).squeeze(-1)
    vertices[...,0] = -vertices[...,0] # -z

    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fix_inversion(mesh)
    return mesh
   
def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n=-1):
    if n != -1:
        vpos, _ = trimesh.sample.sample_surface_even(m, n)
    return torch.tensor(m.vertices, dtype=torch.float32, device="cuda")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chamfer loss')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('-t', type=str, default="diffuse-40-hqq", help="config name")
    parser.add_argument('name', type=str, help="name of the model")
    parser.add_argument('--src_path', type=str, default='', help="src sdf volume or mesh")
    parser.add_argument('--ref_path', type=str, default='', help="ref mesh")

    FLAGS = parser.parse_args()
    if FLAGS.src_path == '':
        FLAGS.src_path = f"outputs/{FLAGS.name}/{FLAGS.t}/warp/params/sdf-data-0511.vol"
    if FLAGS.ref_path == '': 
        FLAGS.ref_path = f"scenes/{FLAGS.name}/{FLAGS.name}.obj"
    
    if FLAGS.src_path[-3:] == 'vol':
        mesh = extract_mesh(FLAGS.src_path)
        mesh.export(FLAGS.src_path.replace('vol','obj'))
        print("mesh exported to ", FLAGS.src_path.replace('vol','obj'))
    else:
        mesh = trimesh.load(FLAGS.src_path)
    mesh = as_mesh(mesh)
    ref = as_mesh(trimesh.load(FLAGS.ref_path))
    
    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale_ref = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    scale_mesh = 10.0 / np.amax(np.amax(mesh.vertices, axis=0) - np.amin(mesh.vertices, axis=0))
    ref.vertices = ref.vertices * scale_ref
    mesh.vertices = mesh.vertices * scale_mesh

    # Sample mesh surfaces
    vpos_mesh = sample_mesh(mesh, FLAGS.n)
    vpos_ref = sample_mesh(ref, FLAGS.n)

    loss,_ = chamfer_distance(vpos_mesh[None, ...], vpos_ref[None, ...])
  
    print("Chamfer Distance: ref [%7d tris -> mesh [%7d tris]: %1.5f" % ((ref.faces.shape[0], mesh.faces.shape[0], loss)))

    opt_dir = Path(FLAGS.src_path).parent.parent / 'opt'

    if opt_dir.exists():
        imgs = [p for p in opt_dir.iterdir() if p.parts[-1].startswith('opt-0511')]
        
        psnrs = []
        for img in imgs:
            id = int(img.parts[-1].split('-')[-1].split('.')[0])
            src_img = cv2.imread(str(img), cv2.IMREAD_COLOR)[...,::-1] / 255.
            fn = str(opt_dir.parent / f'ref-{id:02d}.exr')
            ref_img = read_img(fn)
            mse = np.mean((src_img - ref_img) ** 2)
            mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)
            psnrs.append(mse2psnr(mse))

        print("PSNR: %.2f " % (np.array(psnrs).mean()))    