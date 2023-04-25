import os
import glob
import tqdm
import numpy
import trimesh


for pcf in tqdm.tqdm(glob.glob("scene-pcs/**/*.npy", recursive=True)):
    pc = numpy.load(pcf)
    basename = os.path.splitext(os.path.basename(pcf))[0]
    trimesh.PointCloud(pc).export(f"scene-ply/{basename}.ply")
