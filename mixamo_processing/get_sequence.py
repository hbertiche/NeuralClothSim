import os
import sys
import bpy
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from bpy_utils import init_scene, get_keyframes


def main(file, dst):
    assert os.path.isfile(file), f"File does not exist: {file}"
    assert os.path.isdir(
        os.path.dirname(dst)
    ), f"Folder does not exist: {os.path.dirname(dst)}"

    sc = init_scene()
    # Load FBX
    bpy.ops.import_scene.fbx(filepath=file)

    # Get reference to Armature
    arm_ob = bpy.data.objects["Armature"]

    s, e = get_keyframes(arm_ob)
    sc.frame_start, sc.frame_end = s, e
    """ SKELETON """
    bones = arm_ob.pose.bones
    rbones = arm_ob.data.bones  # unused?

    """ QUATERNIONS AND LOCATIONS """
    Q, L = [], []
    for frame in range(s, e + 1):
        sc.frame_set(frame)
        L += [bones[0].location.copy()]
        q = []
        for b, rb in zip(bones, rbones):
            q += [b.rotation_quaternion]
        Q += [np.stack(q, 0)]
    Q = np.stack(Q, 0)
    L = np.stack(L, 0)

    """ SAVE """
    np.savez(
        dst,
        poses=Q,
        trans=L,
        fps=float(sc.render.fps),
        source=file.split(os.path.sep)[-1],
    )

    sys.exit()


if __name__ == "__main__":
    main(file="path_to_fbx_file.fbx", dst="path_to_output.npz")