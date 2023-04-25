import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AlignToX, GlobalInfoProp, SparseSO3Conv
from time import time


class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False) -> None:
        super().__init__()
        assert(bn is False)
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x + x_res


class PointEncoder(nn.Module):
    def __init__(self, k, spfcs, out_dim) -> None:
        super().__init__()
        self.k = k
        self.spconv = SparseSO3Conv(32, 1, 32, *spfcs)
        self.aggr = GlobalInfoProp(out_dim, out_dim // 4)

    def forward(self, pc, dist):
        nbrs_idx = torch.topk(dist, self.k, largest=False, sorted=False)[1]  #[..., N, K]
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        y = self.spconv(pc_nbrs, pc_nbrs_norm, pc)
        feat = self.aggr(F.relu(y))

        return feat
    
    def forward_nbrs(self, pc, pc_nbrs):
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        y = self.spconv(pc_nbrs, pc_nbrs_norm, pc)
        feat = self.aggr(F.relu(y))

        return feat


class PPFEncoder(nn.Module):
    def __init__(self, ppffcs, out_dim) -> None:
        super().__init__()
        self.res_layers = nn.ModuleList()
        for i in range(len(ppffcs) - 1):
            dim_in, dim_out = ppffcs[i], ppffcs[i + 1]
            self.res_layers.append(ResLayer(dim_in, dim_out, bn=False))
        self.final = nn.Linear(ppffcs[-1], out_dim)

    def forward(self, pc, pc_normal, feat, dist=None, idxs=None):
        if idxs is not None:
            return self.forward_with_idx(pc[0], pc_normal[0], feat[0], idxs)[None]
        xx = pc.unsqueeze(-2) - pc.unsqueeze(-3)
        xx_normed = xx / (dist[..., None] + 1e-7)

        outputs = []
        for idx in torch.chunk(torch.arange(pc.shape[1]), 5):
            feat_chunk = feat[..., idx, :]
            target_shape = [*feat_chunk.shape[:-2], feat_chunk.shape[-2], feat.shape[-2], feat_chunk.shape[-1]]  # B x NC x N x F
            xx_normed_chunk = xx_normed[..., idx, :, :]
            ppf = torch.cat([torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * xx_normed_chunk, -1, keepdim=True), 
                            torch.sum(pc_normal.unsqueeze(-3) * xx_normed_chunk, -1, keepdim=True), 
                            torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * pc_normal.unsqueeze(-3), -1, keepdim=True), 
                            dist[..., idx, :, None]], -1)
            final_feat = torch.cat([feat_chunk[..., None, :].expand(*target_shape), feat[..., None, :, :].expand(*target_shape), ppf], -1)
        
            output = final_feat
            for res_layer in self.res_layers:
                output = res_layer(output)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=-3)
        return self.final(output)

    def forward_with_idx(self, pc, pc_normal, feat, idxs):
        a_idxs = idxs[:, 0]
        b_idxs = idxs[:, 1]
        xy = pc[a_idxs] - pc[b_idxs]
        xy_norm = torch.norm(xy, dim=-1)
        xy_normed = xy / (xy_norm[..., None] + 1e-7)
        ppf = torch.cat([torch.sum(pc_normal[a_idxs] * xy_normed, -1, keepdim=True),
                        torch.sum(pc_normal[b_idxs] * xy_normed, -1, keepdim=True),
                        torch.sum(pc_normal[a_idxs] * pc_normal[b_idxs], -1, keepdim=True),
                        xy_norm[..., None]], -1)
        
        final_feat = torch.cat([feat[a_idxs], feat[b_idxs], ppf], -1)
        
        output = final_feat
        for res_layer in self.res_layers:
            output = res_layer(output)
        return self.final(output)


if __name__ == '__main__':
    
    x = torch.randn(1, 2550, 3).cuda()
    point_encoder = PointEncoder(k=30, spfcs=[32, 64, 32, 32], out_dim=32).cuda()
    ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=2).cuda()

    a = time()
    feat = point_encoder(x)
    ppf_encoder(x, x, feat)
    b = time()
    print(b - a)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy()[0])
    a = time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    b = time()
    print(b - a)