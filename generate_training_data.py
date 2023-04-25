import os
import glob
import tqdm
import numpy
import trimesh
import trimesh.sample


def fromto_cylinder(fr, to, col=None, r=0.02):
    return trimesh.creation.cylinder(r, sections=360, segment=[fr, to], vertex_colors=col)


for obj_path in tqdm.tqdm(glob.glob("scene-pcs/single-object-canonical/*.npy")):
    obj_cls = os.path.basename(obj_path).split("-")[0]
    obj_idx = os.path.splitext(os.path.basename(obj_path))[0].split("-")[1]
    for m_cls in os.listdir(r"D:\SketchPad\scan_form"):
        if m_cls.replace("_", "").replace(" ", "") == obj_cls:
            break
    else:
        raise ValueError("Unknown category", obj_cls, obj_idx)
    obj = trimesh.load(f'D:/SketchPad/scan_form/{m_cls}/{obj_idx}.obj')

    pcd, fidx = trimesh.sample.sample_surface_even(obj, 2560)
    pcd /= 1000.0
    centroid = obj.centroid / 1000.0
    nor = obj.face_normals[fidx]
    # print(centroid)
    # trimesh.Scene([
    #     trimesh.PointCloud(pcd),
    #     fromto_cylinder([0, 0, 0], [0, 0, 1], [0., 0., 1., .5]),
    #     fromto_cylinder([0, 0, 0], [0, 1, 0], [0., 1., 0., .5]),
    #     fromto_cylinder([0, 0, 0], [1, 0, 0], [1., 0., 0., .5]),
    # ]).show(line_settings={"point_size": 5})

    p1 = pcd[:, None]
    p2 = pcd[None, :]
    mu = ((centroid - p1) * (p2 - p1) / (numpy.linalg.norm(p2 - p1, axis=-1, keepdims=True) + 1e-7)).sum(-1)
    nu = numpy.sqrt(numpy.linalg.norm(centroid - p1, axis=-1) ** 2 - mu ** 2)

    numpy.savez(
        f'data/{os.path.splitext(os.path.basename(obj_path))[0]}.npz',
        xyzn=numpy.concatenate([pcd, nor], axis=1).astype(numpy.float32),
        target=numpy.stack([mu, nu], axis=-1).astype(numpy.float32)
    )
