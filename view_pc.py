import numpy
import torch
import trimesh
from torch_redstone import torch_to_numpy
from train_ppf import simplify
import teaserpp


def fromto_cylinder(fr, to, col=None, r=0.02):
    return trimesh.creation.cylinder(r, sections=360, segment=[fr, to], vertex_colors=col)


obj = numpy.load('data/clock-012.npz')['xyzn']
scn = numpy.load('scene-pcs/synthetic-single/syn-1-14.npy')
# scn = numpy.load('scene-pcs/single-object-canonical/clock-012.npy')
# obj, scene, res = torch_to_numpy(torch.load('res.pt'))
T = numpy.eye(4)
T, score = teaserpp.run_full(obj[:, :3], scn, 0.01, True)
print(score)
# T[:3] = res['pose'][-1, 0]
trimesh.Scene([
    trimesh.PointCloud(obj[:, :3], [1.0, 0.2, 0.5, 0.4]),
    trimesh.PointCloud(trimesh.transform_points(scn, numpy.linalg.inv(T)), [0.5, 0.2, 1.0, 0.4])
]).show(line_settings={"point_size": 5})
# pc = numpy.load('scene-pcs/single-object-canonical/hammer-005.npy')
# pc[..., 2] = 0
pc = pc[simplify(pc)]
# pc += numpy.clip(0.005 / 4 * numpy.random.randn(*pc.shape), -0.005 / 2, 0.005 / 2).astype('float32')
trimesh.Scene([
    trimesh.PointCloud(pc),
    fromto_cylinder([0, 0, 0], [0, 0, 1], [0., 0., 1., .5]),
    fromto_cylinder([0, 0, 0], [0, 1, 0], [0., 1., 0., .5]),
    fromto_cylinder([0, 0, 0], [1, 0, 0], [1., 0., 0., .5]),
    trimesh.creation.icosphere(4, radius=0.1, color=[0.8, 0.3, 0.3, 0.5]).apply_translation(pc.mean(0))
]).show(line_settings={"point_size": 5})
