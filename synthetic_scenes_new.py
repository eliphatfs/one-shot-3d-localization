import numpy
import trimesh
import trimesh.sample
import trimesh.creation


obj = trimesh.load(r'I:\SJTU\pc-edit-dist\scan_subset\mug\001.obj')

pcd, fidx = trimesh.sample.sample_surface_even(obj, 2560)
pcd /= 1000.0
centroid = obj.centroid / 1000.0
pcd -= centroid

scene = trimesh.creation.box([1, 0.01, 1]).apply_translation([0, 0.1, 0])
ps, fidx = trimesh.sample.sample_surface_even(scene, 10240)

pcd = numpy.concatenate([ps, pcd])
trimesh.PointCloud(pcd).show(line_settings={'point_size': 4})
numpy.savez('synthscn.npz', xyz=pcd.astype(numpy.float32))
