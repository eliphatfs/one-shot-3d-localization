import os
import json
import tqdm
import glob
import numpy
import torch
import open3d
import trimesh
import trimesh.creation
import trimesh.registration
import scipy.spatial as ssp
import scipy.optimize as sopt
import matplotlib.pyplot as plotlib
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from train_ppf import validation, simplify
from itertools import permutations, product
from models.model import PPFEncoder, PointEncoder


def estimate_normals(pc):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=30))
    return numpy.array(pcd.normals)


def fromto_cylinder(fr, to, col=None, r=0.02):
    return trimesh.creation.cylinder(r, sections=360, segment=[fr, to], vertex_colors=col)


def nms(cands, radius=0.3):
    blocked = []
    for cand in cands:
        if len(blocked) and numpy.linalg.norm(numpy.array(blocked) - cand, axis=-1).min() < radius:
            continue
        blocked.append(cand)
        yield cand


@torch.no_grad()
def predict(xyz, rad_scale, pred_filter=None, vote_res=0.005):
    pc = xyz
    # indices = simplify(pc)
    # pc = pc + numpy.clip(0.004 / 4 * numpy.random.randn(*pc.shape), -0.004 / 2, 0.004 / 2)
    indices = numpy.arange(len(pc))
    high_res_pc = pc[indices].astype(numpy.float32)
    high_res_pc_normal = estimate_normals(high_res_pc).astype(numpy.float32)
    hd_kdt = KDTree(high_res_pc)

    indices = simplify(high_res_pc, 0.025)
    pc = high_res_pc[indices]
    pc_normal = high_res_pc_normal[indices]
    knn_nbrs = high_res_pc[hd_kdt.query(pc, 30, return_distance=False)]

    tree = KDTree(pc)
    nn_idxs = tree.query_radius(pc, rad_scale * 1.5)
    mass_ref2 = tree.query_radius(pc, rad_scale * 2.25, count_only=True).astype(numpy.float32)
    point_idxs = []
    mass = []
    for j, nn_idx in enumerate(nn_idxs):
        point_idxs.append(numpy.stack([numpy.full_like(nn_idx, j), nn_idx], -1))
        mass.append(numpy.full_like(nn_idx, mass_ref2[j] / len(nn_idx)))

    point_idxs = torch.from_numpy(numpy.concatenate(point_idxs, 0)).cuda()
    mass = torch.from_numpy(numpy.concatenate(mass).astype(numpy.float32)).cuda()
    pc = torch.from_numpy(pc).cuda()
    pc_normal = torch.from_numpy(pc_normal).cuda()
    knn_nbrs = torch.from_numpy(knn_nbrs).cuda()

    preds = 0
    num_votes = 1
    for vote in range(num_votes):
        pcv = pc + torch.rand_like(pc) * 0.004 - 0.002
        knn_nbrsv = knn_nbrs + torch.rand_like(knn_nbrs) * 0.004 - 0.002
        sprin_feat = point_encoder.forward_nbrs(pcv[None], knn_nbrsv[None])
        subvote = ppf_encoder.forward_with_idx(pcv, pc_normal, sprin_feat[0], point_idxs).reshape(-1, 2)
        preds = preds + subvote
    preds /= num_votes
    if pred_filter is not None:
        mask = pred_filter(preds)
        point_idxs = point_idxs[mask]
        preds = preds[mask]
        mass = mass[mask]
    # plotlib.hist(preds.cpu().numpy()[:, 1], 50)
    # plotlib.show()
    mu_prob, nu_prob = torch.unbind(preds.reshape(-1, 2, 24), 1)
    mu = grids[0][torch.multinomial(torch.softmax(mu_prob, -1), 1)[..., 0]]
    nu = grids[1][torch.multinomial(torch.softmax(nu_prob, -1), 1)[..., 0]]
    _, cands = validation(
        pc[point_idxs[:, 0]],
        pc[point_idxs[:, 1]],
        # preds[..., 0], preds[..., 1],
        mu, nu,
        vote_res, mass=mass
    )
    return preds.cpu().numpy(), cands


def fib_sphere(n):
    k = numpy.arange(n, dtype=numpy.float32) + 0.5
    phi = numpy.arccos(1 - 2 * k / n)
    theta = numpy.pi * (1 + numpy.sqrt(5)) * k
    x = numpy.cos(theta) * numpy.sin(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(phi)
    return numpy.stack([x, y, z], -1)


def fib_sphere_rot(n):
    stacked = []
    for up in fib_sphere(n):
        right = numpy.cross(numpy.random.randn(3), up)
        right /= numpy.linalg.norm(right)
        fwd = numpy.cross(right, up)
        R = numpy.array([right, up, fwd])
        stacked.append(R)
    return numpy.stack(stacked)


def normalize2(pc1, pc2):
    all_stats = numpy.concatenate([pc1, pc2])
    shift = all_stats.mean(0)
    scale = numpy.linalg.norm(all_stats - shift, axis=-1).max()
    pc1 = (pc1 - shift) / scale
    pc2 = (pc2 - shift) / scale
    return pc1.astype(numpy.float32), pc2.astype(numpy.float32)


def view_cut(pc1, pc2):
    nnd2, i2 = KDTree(pc2).query(pc1, return_distance=True)  # M
    di = pc2[i2[0]].mean(0) - pc2.mean(0)
    proj = pc2.dot(di)
    pc2 = pc2[proj > numpy.mean(proj)]
    return pc2


def iou(pc1, pc2):
    # M x 3, N x 3; scan, ref
    # n1 = estimate_normals(pc1)
    # n2 = estimate_normals(pc2)
    # pc1, pc2 = normalize2(pc1, pc2)
    th = 0.01
    nnd1, i1 = KDTree(pc1).query(pc2, return_distance=True)  # N
    nnd2, i2 = KDTree(pc2).query(pc1, return_distance=True)  # M
    nnd1, i1, nnd2, i2 = nnd1.reshape(-1), i1.reshape(-1), nnd2.reshape(-1), i2.reshape(-1)
    npos = len(pc2)
    fp_sum = numpy.count_nonzero((nnd2 > th))  # | (abs((n1 * n2[i2]).sum(-1)) < numpy.cos(numpy.deg2rad(20))))
    fn_sum = numpy.count_nonzero((nnd1 > th))  # | (abs((n2 * n1[i1]).sum(-1)) < numpy.cos(numpy.deg2rad(20))))
    # trimesh.Scene([
    #     trimesh.PointCloud(pc1, [255, 63, 127]),
    #     trimesh.PointCloud(pc2, [127, 63, 255]),
    # ]).show(line_settings={'point_size': 5})
    return (npos - fn_sum) / (npos + fp_sum)


def sim_pn(pc1, pc2):
    from pnlkrev import pn
    pc1, pc2 = normalize2(pc1, pc2)
    F = torch.nn.functional
    with torch.no_grad():
        pn.eval()
        f1 = pn(torch.from_numpy(pc1[None]).float().cuda(), 0)
        f2 = pn(torch.from_numpy(pc2[None]).float().cuda(), 0)
    return 1 - F.l1_loss(f1, f2).item()


def sim_df(pc1, pc2):
    pc1, pc2 = normalize2(pc1, pc2)
    q = numpy.random.randn(8192, 3)
    q = q / numpy.linalg.norm(q, axis=-1, keepdims=True) * numpy.random.rand(len(q), 1) ** (1 / 2)
    q = q.astype(numpy.float32)
    nnd1, i1 = KDTree(pc1).query(q, return_distance=True)  # N
    nnd2, i2 = KDTree(pc2).query(q, return_distance=True)  # M
    nnd1, i1, nnd2, i2 = nnd1.reshape(-1), i1.reshape(-1), nnd2.reshape(-1), i2.reshape(-1)
    return 1 - abs(nnd1 - nnd2).mean()


def registration(pc, pc_ref, return_iou=False):
    max_iou, max_simdf, max_res = -1, -1, None
    t = pc.mean(0) - pc_ref.mean(0)
    tpc = pc - t
    Rs = fib_sphere_rot(50)
    # d1 = KDTree(pc).query(pc, k=2, return_distance=True)[0][:, 1]
    # d2 = KDTree(pc_ref).query(pc_ref, k=2, return_distance=True)[0][:, 1]
    # len_max = numpy.arange(max(len(d1), len(d2)) * 3)
    # iouth = (numpy.percentile(numpy.concatenate([d1[len_max % len(d1)], d2[len_max % len(d2)]]), 96))
    # iouth = max(0.005, numpy.percentile(d2, 95))
    # print(iouth)
    # plotlib.style.use('seaborn')
    # plotlib.hist(d1, alpha=0.5, label='scan', bins=36)
    # plotlib.hist(d2, alpha=0.5, label='ref', bins=36)
    # plotlib.legend()
    # plotlib.show()
    for R in Rs:
        Rx = numpy.eye(4)
        Rx[:3, :3] = R
        G, _, cost = trimesh.registration.icp(tpc, pc_ref, Rx, scale=False)
        G = trimesh.transformations.inverse_matrix(G)
        tx = trimesh.transform_points(pc_ref, G)
        miou = iou(tpc, tx)
        simdf = sim_df(tpc, tx)
        # print(miou, cost)
        if simdf > max_simdf:
            max_iou = miou
            max_simdf = simdf
            max_res = trimesh.transformations.translation_matrix(t) @ G
    # print(max_iou)
    if return_iou:
        return max_iou, max_simdf, max_res
    else:
        return max_res


def run_scene(obj_name, scene, anno):
    ckpt = torch.load(f'ckpt/{obj_name}.pt')
    point_encoder.load_state_dict(ckpt['pe'])
    ppf_encoder.load_state_dict(ckpt['ppf'])
    grids.clear()
    grids.extend([ckpt['mu_grid'].cuda(), ckpt['nu_grid'].cuda()])
    point_encoder.cuda().eval()
    ppf_encoder.cuda().eval()

    obj_pts = numpy.load(f'data/{obj_name}.npz')['xyzn'][:, :3]
    obj_pts -= obj_pts.mean(0)
    rad_scale = numpy.linalg.norm(obj_pts, axis=-1).max()
    # print(rad_scale)
    preds, cands = predict(obj_pts, rad_scale)
    clean_err = numpy.linalg.norm(cands[0])
    # print("Clean:", clean_err)
    low = numpy.percentile(preds, 1, axis=0)
    high = numpy.percentile(preds, 99, axis=0)
    xyz = numpy.load(scene).astype(numpy.float32)[:, [0, 2, 1]]
    # xyz = numpy.load(r'synthscn.npz')['xyz'].astype(numpy.float32)
    # xyz = obj_pts
    _, cands = predict(
        xyz, rad_scale,
        # lambda preds: ((preds > preds.new_tensor(low)) & (preds < preds.new_tensor(high))).all(-1),
        vote_res=0.01
    )
    anno = anno['center']
    gt = numpy.array([anno['x'], anno['z'], anno['y']])
    # gt = numpy.zeros(3)
    minrank = [999] * tops.shape[-1]
    proposals = []
    for i, cand in enumerate(nms(cands)):
        if i == 10:
            break
        for icm, cm in enumerate(cms):
            if numpy.linalg.norm(cand - gt) < cm * 0.01:
                minrank[icm] = min(minrank[icm], i)
        if i >= 5:
            continue
        tree = KDTree(xyz)
        nn_idxs = tree.query_radius([cand], rad_scale + 0.01)[0]
        if len(nn_idxs) == 0:
            continue
        miou, simdf, G = registration(xyz[nn_idxs], obj_pts, return_iou=True)
        proposals.append((miou, simdf, trimesh.transform_points(obj_pts, G), nn_idxs))
        # proposals.append((miou, trimesh.transform_points(obj_pts, G).mean(0)))
        
    # scale_iou = numpy.ptp([x[0] for x in proposals])
    # scale_simdf = numpy.ptp([x[1] for x in proposals])
    max_iou, max_simdf, f, nn_idxs = max(proposals, key=lambda x: x[1])
    # max_iou, localization = max(proposals, key=lambda x: x[0])
    print()
    for miou, sdf, tx, _ in proposals:
        loc = tx.mean(0)
        print('%.2f %.2f %.2f' % (miou * 100, sdf * 100, numpy.linalg.norm(loc - gt) * 100))
    localization = f.mean(0)
    print('%.2f %.2f %.2f' % (max_iou * 100, max_simdf * 100, numpy.linalg.norm(localization - gt) * 100))
    print(scene, obj_name)
    # vis_seg_nn = numpy.zeros(len(xyz), dtype=numpy.int8)
    # vis_seg_nn[nn_idxs] = 1
    # trimesh.Scene([
    #     trimesh.PointCloud(
    #         xyz,
    #         # colors=numpy.exp(-numpy.linalg.norm(xyz - gt, axis=-1, keepdims=True) / 0.15) * [1., 0., 0.]
    #         colors=numpy.array([[0, 0, 0], [255, 60, 60]])[vis_seg_nn]
    #     ),
    #     trimesh.PointCloud(f, [0.6, 0.1, 1.0, 0.5]),
    #     # fromto_cylinder([0, 0, 0], [0, 0, 1], [0., 0., 1., .5]),
    #     # fromto_cylinder([0, 0, 0], [0, 1, 0], [0., 1., 0., .5]),
    #     # fromto_cylinder([0, 0, 0], [1, 0, 0], [1., 0., 0., .5]),
    #     # trimesh.creation.icosphere(4, radius=0.1, color=[0.8, 0.3, 0.3, 0.5]).apply_translation(gt)
    # ]).show(line_settings={'point_size': 4})
    for icm, cm in enumerate(cms):
        if numpy.linalg.norm(localization - gt) < cm * 0.01:
            accs[icm] += 1
    for ik, k in enumerate(ks):
        for icm, r in enumerate(minrank):
            if r < k:
                tops[ik, icm] += 1

# mug_001 2023-03-22/11-25-13 2023-03-20/14-23-40
# clock_001 2023-03-19/21-27-36
class_to_obj = [
    'bowl-018',
    'clock-012',
    'container-003',
    'hammer-005',
    'handbag-013',
    'hat-003',
    'mug-001',
    'ramenbox-020',
    'remote-004',
    'sausage-003',
    'tennisball-002',
    'thermos-002',
    'toothbrush-029',
    'toycar-014',
    'trashbin-003',
    # extra
    'bowl-013',
    'mug-002'
]
obj_name = 'bowl-013'
tops = numpy.zeros([3, 3])
accs = numpy.zeros([3])
ks = [1, 5, 10]
cms = [5, 10, 20]

point_encoder = PointEncoder(k=30, spfcs=[32, 64, 32, 32], out_dim=32)
ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=48)
grids = []

count = 0
for obj_name in tqdm.tqdm(class_to_obj):
    scene = f'scene-pcs/single-object-canonical/{obj_name}.npy'
    with open(scene.replace('.npy', '.json')) as fi:
        anno = json.load(fi)['items'][0]
    run_scene(obj_name, scene, anno)
    count += 1
# prog = tqdm.tqdm(glob.glob('scene-pcs/**/*.npy', recursive=True))
# for scene in prog:
#     prog.set_description(os.path.splitext(os.path.basename(scene))[0])
#     with open(scene.replace('.npy', '.json')) as fi:
#         annos = json.load(fi)['items']
#     for anno in annos:
#         if len([x for x in annos if x['kind'] == anno['kind']]) >= 2:
#             continue
#         count += 1
#         run_scene(class_to_obj[anno['kind']], scene, anno)


with open("eval_new.txt", "w") as fo:
    print("All:", count, file=fo)
    for ik, k in enumerate(ks):
        for icm, cm in enumerate(cms):
            print("%dcm@%d: %.2f" % (cm, k, tops[ik, icm] / count * 100), file=fo)
    for icm, cm in enumerate(cms):
        print("Acc@%dcm: %.2f" % (cm, accs[icm] / count * 100), file=fo)
print(open("eval_new.txt").read())
