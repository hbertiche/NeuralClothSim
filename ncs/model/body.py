import os
import json
import numpy as np
import tensorflow as tf

from utils.rotation import from_axis_angle, from_quaternion
from utils.tensor import tf_shape

from global_vars import BODY_DIR
from utils.mesh import triangulate


class Body:
    def __init__(self, body_model, input_joints=None):
        self.body_model = body_model
        # Read body model
        with np.load(body_model) as model:
            self.vertices = model["vertices"]
            self.faces = triangulate(model["faces"])
            self.blend_weights = model["blend_weights"].astype(np.float32)
            self.joints = np.float32(model["joints"])
            self.rest_pose = model["rest_pose"]
            self.parents = model["parents"]
            if "no_collide_vertices" in model:
                self.no_collide_vertices = model["no_collide_vertices"]
                assert not set(self.collision_vertices) == set(
                    range(self.num_verts)
                ), "Error! There is no collision vertices."
        self.input_joints = input_joints or list(range(self.num_joints))
        self.tree_to_depth()
        # Infer rotation mode ('axis_angle' or 'quaternion')
        n_dims = self.rest_pose.shape[1]
        if n_dims == 3:
            self.rotation_mode = "axis_angle"
        elif n_dims == 4:
            self.rotation_mode = "quaternion"

        # Compute local matrices (rest pose rotations)
        self.local_matrices()

        # Create 4x4 matrix (for 3x3 to 4x4 function)
        self._to4x4_aux = np.zeros((4, 4), np.float32)
        self._to4x4_aux[3, 3] = 1

    @property
    def num_joints(self):
        return self.joints.shape[0]

    @property
    def num_input_joints(self):
        return len(self.input_joints)

    @property
    def num_verts(self):
        return self.vertices.shape[0]

    @property
    def no_collide(self):
        return hasattr(self, "no_collide_vertices")

    @property
    def collision_vertices(self):
        indices = list(range(self.num_verts))
        if not self.no_collide:
            return indices
        return list(set(indices) - set(self.no_collide_vertices))

    def tree_to_depth(self):
        root = np.nonzero(self.parents < 0)[0].tolist()
        assert (
            len(root) == 1
        ), "Body model error. No root joint or multiple root joints: " + str(root)
        root = root[0]
        depth = [[root]]
        visited = set()
        visited.add(root)
        while len(visited) < len(self.parents):
            children = sum(
                [
                    np.nonzero(self.parents == parent)[0].tolist()
                    for parent in depth[-1]
                ],
                [],
            )
            depth += [children]
            visited.update(children)
        self.depth = depth[1:]

    def local_matrices(self):
        self.rotations_local = np.zeros((self.num_joints, 4, 4), np.float32)
        self.rotations_local_inv = np.zeros(self.rotations_local.shape, np.float32)
        if self.rotation_mode == "axis_angle":
            rotations = from_axis_angle(self.rest_pose).numpy()
        elif self.rotation_mode == "quaternion":
            rotations = from_quaternion(self.rest_pose).numpy()
        for i in range(self.num_joints):
            self.rotations_local[i, :3, :3] = rotations[i]
            self.rotations_local[i, :, 3] = *self.joints[i], 1
            self.rotations_local_inv[i] = np.linalg.inv(self.rotations_local[i])

    @tf.function
    def forward_kinematics(self, rotations_basis, root_locations):
        batch_shape = tf_shape(rotations_basis)[:-3]
        rotations_basis = tf.reshape(rotations_basis, (-1, self.num_joints, 3, 3))
        root_locations = tf.reshape(root_locations, (-1, 3))
        rotations = [None] * self.num_joints
        rotations_basis = self.to4x4(rotations_basis)

        rotations_basis += self.root_loc_to_4x4(root_locations)

        rotations[0] = tf.einsum(
            "ab,cbd->cad", self.rotations_local[0], rotations_basis[:, 0]
        )
        for depth in self.depth:
            parents = self.parents[depth]
            rotation_parents = tf.stack(
                [rotations[parent] for parent in parents], axis=1
            )
            rot = tf.einsum(
                "abcd,bde->abce", rotation_parents, self.rotations_local_inv[parents]
            )
            rot = tf.einsum("abcd,bde->abce", rot, self.rotations_local[depth])
            rot = tf.einsum(
                "abcd,abde->abce", rot, tf.gather(rotations_basis, depth, axis=1)
            )
            for i, n in enumerate(depth):
                rotations[n] = rot[:, i]

        rotations = tf.stack(rotations, axis=1)
        rotations = tf.einsum("abcd,bde->abce", rotations, self.rotations_local_inv)[
            :, :, :3
        ]
        return tf.reshape(rotations, (*batch_shape, self.num_joints, 3, 4))

    @tf.function
    def to4x4(self, matrix):
        pad = [[0, 0] for n in range(len(matrix.shape) - 2)]
        pad += [[0, 1], [0, 1]]
        matrix = tf.pad(matrix, pad)
        matrix += self._to4x4_aux
        return matrix

    @tf.function
    def root_loc_to_4x4(self, root_loc):
        pad = [[0, 0], [0, self.num_joints - 1], [0, 1], [3, 0]]
        return tf.pad(root_loc[:, None, :, None], pad)
