import os
import sys

import bpy
import numpy as np

sys.path.append(os.path.dirname(__file__))
from bpy_utils import init_scene, decomposeObject, getVertexGroups
from utils import merge_meshes, triangulate


def main(file, dst):
    assert os.path.isfile(file), f"FBX file does not exist: {file}"
    assert os.path.isdir(
        os.path.dirname(dst)
    ), f"Save folder does not exist: {os.path.dirname(dst)}"

    sc = init_scene()
    # Load FBX
    bpy.ops.import_scene.fbx(filepath=file)

    # Get reference to 'Armature' (skeleton)
    arm_ob = bpy.data.objects["Armature"]
    # Get reference of body (may be more than one object)
    obs = [ob for ob in bpy.data.objects if ob.name != "Armature"]

    """ SKELETON """
    bones = arm_ob.pose.bones
    rbones = arm_ob.data.bones
    # Kinematic tree
    bnames = [b.name for b in bones]
    parents = [bnames.index(b.parent.name) if b.parent else -1 for b in bones]
    # Rest pose skeleton
    joints = np.array([rb.matrix_local.to_translation() for rb in rbones], np.float32)
    rest_pose = np.array([rb.matrix_local.to_quaternion() for rb in rbones], np.float32)
    """ TEMPLATE AND BLEND WEIGHTS """
    vertices, faces, blend_weights = [], [], []
    for ob in obs:
        me = ob.to_mesh()
        vertices += [np.float32([v.co for v in me.vertices])]
        faces += [decomposeObject(ob)[1]]
        blend_weights += [getVertexGroups(ob, bones)]
    vertices, faces = merge_meshes(vertices, faces)
    faces = triangulate(faces)
    blend_weights = np.concatenate(blend_weights, axis=0)
    """ Scale down (Mixamo scale is x100)"""
    vertices *= 0.01
    joints *= 0.01
    """ Store """
    np.savez(
        dst,
        vertices=vertices,
        faces=faces,
        blend_weights=blend_weights,
        joints=joints,
        rest_pose=rest_pose,
        parents=parents,
    )
    sys.exit()


if __name__ == "__main__":
    main(file="path_to_fbx_file.fbx", dst="path_to_output.npz")