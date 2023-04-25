import numpy
import trimesh
import matplotlib.pyplot as plotlib
from scipy.spatial.distance import cdist

from pnlkrev import pointnet_lk_registration
pc = numpy.load(f'data/mug-002.npz')['xyzn'][:, :3]
pc -= pc.mean(0)
pc /= numpy.linalg.norm(pc, axis=-1).max()


def iou(pc1, pc2, th=0.02):
    dist = cdist(pc1, pc2)
    npos = len(numpy.min(dist, -2))
    fp_sum = numpy.count_nonzero(numpy.min(dist, -1) > th)
    fn_sum = numpy.count_nonzero(numpy.min(dist, -2) > th)
    return (npos - fn_sum) / (npos + fp_sum)


rots = [15, 30, 45, 60, 75, 90, 105, 120, 135]
errs = []
stds = []
deg_err = []
all_deg = []
reg_iou = []
for rot in rots:
    for t in range(10):
        rax = numpy.random.randn(3)
        rax /= numpy.linalg.norm(rax)
        R = trimesh.transformations.rotation_matrix(numpy.deg2rad(rot), rax)[:3, :3]
        g, f = pointnet_lk_registration(pc @ R, pc)
        g = g[0].cpu().numpy()
        deg_err.append(numpy.rad2deg(numpy.arccos(numpy.clip((numpy.trace(g[:3, :3] @ R) - 1) / 2, 0, 1))))
        all_deg.append(deg_err[-1])
        reg_iou.append(iou(pc @ R, f[0].cpu().numpy().T))
        # deg_err.append(numpy.rad2deg(numpy.max(numpy.arccos(numpy.clip((R.T * g[:3, :3]).sum(1), 0, 1)))))
    errs.append(numpy.mean(deg_err))
    stds.append(numpy.std(deg_err))
    deg_err.clear()
plotlib.style.use('seaborn')
plotlib.subplot(211)
plotlib.errorbar(rots, errs, stds)
plotlib.subplot(212)
plotlib.scatter(all_deg, reg_iou)
plotlib.show()
