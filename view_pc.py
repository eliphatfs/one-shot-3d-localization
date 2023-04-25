import numpy
import trimesh
from train_ppf import simplify


def fromto_cylinder(fr, to, col=None, r=0.02):
    return trimesh.creation.cylinder(r, sections=360, segment=[fr, to], vertex_colors=col)


pc = numpy.load('data/hammer-005.npz')['xyzn'][:, :3]
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
