import os
import tqdm
import glob
import uuid
import json
import numpy
import random
import trimesh
import trimesh.sample
import trimesh.creation
from typing import List


obj_list: List[trimesh.Trimesh] = []
obj_ids = []
obj_kinds = []


def load_all_objects():
    for obj_path in tqdm.tqdm(glob.glob("scene-pcs/single-object-canonical/*.npy")):
        obj_cls = os.path.basename(obj_path).split("-")[0]
        obj_idx = os.path.splitext(os.path.basename(obj_path))[0].split("-")[1]
        for m_cls in os.listdir(r"D:\SketchPad\scan_form"):
            if m_cls.replace("_", "").replace(" ", "") == obj_cls:
                break
        else:
            raise ValueError("Unknown category", obj_cls, obj_idx)
        obj: trimesh.Trimesh = trimesh.load(f'D:/SketchPad/scan_form/{m_cls}/{obj_idx}.obj')
        obj.apply_scale(0.001)
        obj.apply_obb()
        obj_ids.append(obj_idx)
        obj_kinds.append(obj_cls)
        obj_list.append(obj)


def generate_scene(fp, index_list):
    scene = trimesh.creation.box([0.75, 0.01, 0.75])
    obj_list_c = [obj_list[i].copy() for i in index_list]
    cat_list = [scene] + obj_list_c
    collision = trimesh.collision.CollisionManager()
    for obj in obj_list_c:
        while True:
            x, z = numpy.random.uniform(-0.37, 0.37, size=[2])
            obj.apply_translation(numpy.array([x - obj.centroid[0], 0.005 - obj.bounds[0, 1], z - obj.centroid[2]]))
            if not collision.in_collision_single(obj):
                break
        collision.add_object(uuid.uuid4().hex, obj.convex_hull)
    ps, fidx = trimesh.sample.sample_surface_even(trimesh.util.concatenate(cat_list), 25600)
    # trimesh.PointCloud(ps).show(line_settings={'point_size': 4})
    annotation = {
        "kind": "detection",
        "items": [{
            "box": obj_list_c[i].bounds.tolist(),
            "kind": obj_kinds[j],
            "idx": obj_ids[j],
            "center": {
                "x": obj_list_c[i].centroid[0].item(),
                "y": obj_list_c[i].centroid[1].item(),
                "z": obj_list_c[i].centroid[2].item()
            }
        } for i, j in enumerate(index_list)]
    }
    with open(fp + ".json", "w") as fo:
        json.dump(annotation, fo, indent=4)
    numpy.save(fp + '.npy', numpy.array(ps))


load_all_objects()
for i in tqdm.trange(17):
    generate_scene("scene-pcs/synthetic-single/syn-1-%02d" % i, [i])
for i, oc in enumerate(tqdm.tqdm(set(obj_kinds))):
    candidates = [x for x in range(len(obj_list)) if obj_kinds[x] == oc]
    generate_scene(
        "scene-pcs/synthetic-single-category/syn-2-%02d-0" % i,
        [random.choice(candidates) for _ in range(2)]
    )
    generate_scene(
        "scene-pcs/synthetic-single-category/syn-2-%02d-1" % i,
        [random.choice(candidates) for _ in range(5)]
    )
for i in tqdm.trange(2, 9):
    for j in range(3):
        generate_scene(
            "scene-pcs/synthetic-multiple/syn-3-%02d-%d" % (i, j),
            [random.choice(range(len(obj_list))) for _ in range(i)]
        )
