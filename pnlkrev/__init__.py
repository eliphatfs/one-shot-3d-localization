import os
import torch
from .model import AnalyticalPointNetLK, Pointnet_Features as PointNetEncoder
from .utils import transform


pn = PointNetEncoder()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_lk = AnalyticalPointNetLK(pn, device)
model_lk.load_state_dict(torch.load(os.path.join(__file__, "..", "model_trained_on_ModelNet40_model_best.pth"), map_location=device))
model_lk.to(device).eval()


def pointnet_lk_registration(p0, p1):
    p0 = torch.from_numpy(p0[None]).to(device)
    p1 = torch.from_numpy(p1[None]).to(device)
    AnalyticalPointNetLK.do_forward(model_lk, p0, None, p1, None, 10, 1e-7, True, True, 'test')
    g = model_lk.g
    f = transform(g, p1)
    return g, f
