import os
import ctypes
import numpy
import open3d
from sklearn.neighbors import KDTree


def preprocess_point_cloud(pc, voxel_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def basic_matching(descriptors, ref_descriptors):
    """
    Matching strategy that matches each descriptor with its nearest neighbor in the feature space.

    Args:
        descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.

    Returns:
        Indices of the matches established.
    """
    return KDTree(ref_descriptors).query(descriptors, return_distance=False)


# input("...")
for search in os.environ['PATH'].split(os.pathsep):
    if os.path.exists(search):
        os.add_dll_directory(search)
teaser = ctypes.CDLL(os.path.abspath(os.path.join(os.path.dirname(__file__), "teaser.dll")))
teaser_run = teaser.run
teaser_run.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
teaser_run.restype = ctypes.c_int


def run(src_xyzn, tgt_xyzn, return_score=False):
    g = numpy.zeros([4, 4], numpy.float32)
    score = teaser_run(
        len(src_xyzn), len(tgt_xyzn),
        ctypes.c_void_p(src_xyzn.ctypes.data),
        ctypes.c_void_p(tgt_xyzn.ctypes.data),
        ctypes.c_void_p(g.ctypes.data),
    )
    if return_score:
        return g, score
    return g


def run_full(src_xyz, tgt_xyz, voxel_size=0.02, return_score=False):
    src, src_fpfh = preprocess_point_cloud(src_xyz, voxel_size)
    tgt, tgt_fpfh = preprocess_point_cloud(tgt_xyz, voxel_size)
    matching = basic_matching(numpy.array(src_fpfh.data).T, numpy.array(tgt_fpfh.data).T)
    src_xyzn = numpy.concatenate([src.points, src.normals], axis=-1).astype(numpy.float32)
    tgt_xyzn = numpy.concatenate([tgt.points, tgt.normals], axis=-1).astype(numpy.float32)
    tgt_xyzn = tgt_xyzn[matching.reshape(-1)]
    return run(src_xyzn, tgt_xyzn, return_score=return_score)


# arr1 = numpy.random.randn(5000, 3).astype(numpy.float32)
# arr2 = numpy.concatenate([arr1, arr1]) + 1
# print(run_full(arr1, arr2))
