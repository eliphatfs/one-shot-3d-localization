import os
import glob
import math
import hydra
import torch
import open3d
from models.utils import smooth_l1_loss
from models.model import PPFEncoder, PointEncoder
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import visdom
import itertools


vis = visdom.Visdom(port=8097)


def estimate_normals(pc):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=30))
    return np.array(pcd.normals, dtype=np.float32)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pc, target):
        super().__init__()
        self.pc = pc[:, :3]
        self.pc_normal = estimate_normals(pc[:, :3])
        self.target = target
        self.mu_grid = MiniBatchKMeans(24, batch_size=4096).fit(target[..., 0].reshape(-1, 1)).cluster_centers_.reshape(-1).astype(np.float32)
        self.nu_grid = MiniBatchKMeans(24, batch_size=4096).fit(target[..., 1].reshape(-1, 1)).cluster_centers_.reshape(-1).astype(np.float32)
        mu_target = abs(target[..., :1] - self.mu_grid).argmin(-1)
        nu_target = abs(target[..., 1:] - self.nu_grid).argmin(-1)
        self.target = np.stack([mu_target, nu_target], -1)
        self.indices = simplify(self.pc)

    def __getitem__(self, idx):
        # pc = self.pc + np.clip(0.005 / 4 * np.random.randn(*self.pc.shape), -0.005 / 2, 0.005 / 2).astype(np.float32)
        pc = self.pc + np.random.uniform(-0.002, 0.002, size=self.pc.shape).astype(np.float32)
        self.indices = np.random.permutation(len(pc))[:np.random.randint(len(pc) // 3, len(pc) * 4 // 5)]
        normals = estimate_normals(pc[self.indices])
        return [pc[self.indices], normals, self.target[self.indices][:, self.indices]]

    def __len__(self):
        return 50


def simplify(xyz, res=0.005):
    xyz_voxelized = np.round(xyz / res)
    core, indices = np.unique(xyz_voxelized, axis=0, return_index=True)
    return indices


def validation(p1, p2, mu, nu, res, num_rots=72, mass=None):

    corners = torch.stack([torch.min(torch.minimum(p1, p2), 0)[0], torch.max(torch.maximum(p1, p2), 0)[0]]).cpu().numpy()
    grid_res = (((corners[1] - corners[0]) / res).astype(np.int32) + 1).tolist()
    grid_obj = torch.zeros(grid_res, dtype=torch.float32).reshape(-1).to(mu)

    adaptive_n_rots = torch.clamp(nu / res, 1, num_rots).long()
    ab = p2 - p1
    c: torch.Tensor = p1 + ab / torch.norm(ab, dim=-1, keepdim=True) * mu[:, None]
    x = torch.cross(ab, torch.randn_like(ab))
    y = torch.cross(ab, x)
    x = x / torch.norm(x, dim=-1, keepdim=True) * nu[:, None]
    y = y / torch.norm(y, dim=-1, keepdim=True) * nu[:, None]
    for i in range(num_rots):
        angle = i * 2 * torch.pi / adaptive_n_rots
        out_mask = i < adaptive_n_rots
        angle = angle[out_mask]
        if mass is not None:
            mass_masked = mass[out_mask]
        offset = torch.cos(angle.unsqueeze(-1)) * x[out_mask] + torch.sin(angle.unsqueeze(-1)) * y[out_mask]
        center_grid = (c[out_mask] + offset - c.new_tensor(corners[0])) / res
        residual = torch.frac(center_grid)
        for delta in itertools.product([0, 1], [0, 1], [0, 1]):
            delta = c.new_tensor(list(delta))
            ixyz = torch.floor(center_grid + delta).long()
            mask = ((ixyz >= 0) & (ixyz < ixyz.new_tensor(grid_res))).all(-1)
            ixyz = ixyz[mask]
            ix, iy, iz = torch.unbind(ixyz, -1)
            dw = torch.prod(1 - delta - residual, dim=1).abs()[mask]
            if mass is not None:
                dw = dw / mass_masked[mask]
            grid_obj.scatter_add_(
                0,
                (ix * grid_res[1] * grid_res[2] + iy * grid_res[2] + iz),
                dw
            )

    asort = torch.argsort(grid_obj.flatten())
    asort = asort[len(asort) * 2 // 3:].cpu().numpy()
    grid_obj = grid_obj.reshape(grid_res).cpu().numpy()
    vis.heatmap(grid_obj.max(0), win=3, opts=dict(
        title='front'
    ))
    vis.heatmap(grid_obj.max(1), win=4, opts=dict(
        title='bird'
    ))
    vis.heatmap(grid_obj.max(2), win=5, opts=dict(
        title='side'
    ))

    cand = np.array(np.unravel_index(asort, grid_obj.shape)).T
    cand_world = corners[0] + cand * res
    # print(cand_world[-1])
    return grid_obj, cand_world[::-1]


class LR(object):
    def assign_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def warmup_lr(self, base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def __init__(self, optimizer, steps) -> None:
        self.opt = optimizer
        self.t = 0
        self.steps = steps
        self.base_lr = 5e-2
        print("Base LR:", self.base_lr)
        self.step()

    def step(self):
        self.t += 1
        base_lr = self.base_lr
        step = self.t
        warmup_length = 100
        if step < warmup_length:
            lr = self.warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = self.steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        self.assign_learning_rate(self.opt, lr)
        return lr


def train(cfg, data_file):
    logger = logging.getLogger(__name__)
    npz = np.load(hydra.utils.to_absolute_path(data_file))
    vns = npz['xyzn']
    targets = npz['target']

    ds = SimpleDataset(vns, targets)
    df = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    
    point_encoder = PointEncoder(k=30, spfcs=[32, 64, 32, 32], out_dim=32).cuda()
    ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=48).cuda()

    opt = optim.AdamW([*point_encoder.parameters(), *ppf_encoder.parameters()], lr=3e-3, weight_decay=0.2)
    sched = LR(opt, 16 * math.ceil(len(ds) / cfg.batch_size))
    crit = nn.CrossEntropyLoss()
    
    logger.info('Train')
    for epoch in range(cfg.max_epoch):
        n = 0
        running_loss = 0.0
        point_encoder.train()
        ppf_encoder.train()
        mu_grid = torch.tensor(ds.mu_grid).cuda()
        nu_grid = torch.tensor(ds.nu_grid).cuda()
        with tqdm(df) as t:
            for pcs, pc_normals, targets in t:
                pcs, pc_normals, targets = pcs.cuda(), pc_normals.cuda(), targets.cuda()
                opt.zero_grad()
                
                dist = torch.cdist(pcs, pcs)
                sprin_feat = point_encoder(pcs, dist)
                preds = ppf_encoder(pcs, pc_normals, sprin_feat, dist).reshape(-1, 24)
                loss = crit(preds, targets.cuda().reshape(-1))
                # loss = smooth_l1_loss(preds, targets, beta=cfg.res)
                loss.backward()
                running_loss += loss.item()
                n += 1

                opt.step()
                sched.step()

                t.set_postfix(loss=running_loss / n)

            # validation
            point_encoder.eval()
            ppf_encoder.eval()
            pcs, pc_normals, targets = iter(t).__next__()
            pcs, pc_normals, targets = pcs.cuda(), pc_normals.cuda(), targets.cuda()
            with torch.no_grad():
                dist = torch.cdist(pcs, pcs)
                sprin_feat = point_encoder(pcs, dist)
                preds = ppf_encoder(pcs, pc_normals, sprin_feat, dist)
                mu_prob, nu_prob = torch.unbind(preds.reshape(-1, 2, 24), 1)
                mu = mu_grid[torch.multinomial(torch.softmax(mu_prob, -1), 1)[..., 0]]
                nu = nu_grid[torch.multinomial(torch.softmax(nu_prob, -1), 1)[..., 0]]
                xv, yv = torch.meshgrid(torch.arange(pcs.shape[1]), torch.arange(pcs.shape[1]), indexing='xy')
                point_idxs = torch.stack([yv, xv], -1).reshape(-1, 2).to(pcs.device)
                validation(
                    pcs[0, point_idxs[:, 0]],
                    pcs[0, point_idxs[:, 1]],
                    mu, nu,
                    cfg.res
                )
        torch.save({
            'pe': point_encoder.state_dict(),
            'ppf': ppf_encoder.state_dict(),
            'mu_grid': mu_grid,
            'nu_grid': nu_grid
        }, hydra.utils.to_absolute_path('ckpt/' + os.path.basename(data_file).replace('.npz', '.pt')))
        # torch.save(ppf_encoder.state_dict(), f'ppf_encoder_ckpt.pth')
        logger.info('loss: {:.4f}'.format(running_loss / n))


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    for data in glob.glob(hydra.utils.to_absolute_path('data/*.npz')):
        train(cfg, data)


if __name__ == '__main__':
    main()
