import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from utils.IO import readOBJ
from utils.mesh import (
    dihedral_angle_adjacent_faces,
    edge_lengths,
    face_normals,
    laplacian_matrix,
    vertex_area,
    faces_to_edges_and_adjacency,
    triangulate,
)


class Garment:
    def __init__(self, obj):
        if not obj.endswith(".obj"):
            obj += ".obj"
        self.obj = obj
        print("\tReading OBJ file...")
        self.vertices, faces = readOBJ(obj)[:2]
        print("\tTriangulating faces...")
        self.faces = triangulate(faces)
        print("\tComputing face adjacency...")
        (
            self.edges,
            self.face_adjacency,
            self.face_adjacency_edges,
        ) = faces_to_edges_and_adjacency(self.faces)
        print("\tComputing mesh laplacian...")
        self.laplacian = laplacian_matrix(self.faces)
        print("\tComputing edge lengths...")
        self.edge_lengths = edge_lengths(self.vertices, self.edges)
        self.face_adjacency_edge_lengths = edge_lengths(
            self.vertices, self.face_adjacency_edges
        )
        print("\tComputing face normals...")
        self.normals = face_normals(self.vertices, self.faces).numpy()
        print("\tComputing adjacent face dihedral angles...")
        self.face_dihedral = dihedral_angle_adjacent_faces(
            self.normals, self.face_adjacency
        )
        print("\tComputing vertex and total area...")
        self.vertex_area, self.face_areas, self.surf_area = vertex_area(
            self.vertices, self.faces
        )
        print("\tComputing triangles in UV plane")
        self.make_continuum()
        pinning_data = obj.replace(".obj", "_pin.npy")
        if os.path.isfile(pinning_data):
            self.pinning_vertices = np.load(pinning_data).reshape(-1).astype(np.int32)

    @property
    def num_verts(self):
        return self.vertices.shape[0]

    @property
    def pinning(self):
        return hasattr(self, "pinning_vertices")

    def transfer_blend_weights(self, body):
        tree = cKDTree(body.vertices)
        _, idx = tree.query(self.vertices, workers=-1)
        self.blend_weights = body.blend_weights[idx][:, body.input_joints]
        self.blend_weights /= self.blend_weights.sum(1, keepdims=True)

    def smooth_blend_weights(self, iterations=50):
        for _ in range(iterations):
            self.blend_weights = self.laplacian @ self.blend_weights
        self.blend_weights = self.blend_weights.astype(np.float32)

    def make_continuum(self):
        angle = np.arccos(self.normals[:, 2])
        axis = np.stack(
            [
                self.normals[:, 1],
                -self.normals[:, 0],
                np.zeros((self.normals.shape[0],), np.float32),
            ],
            axis=-1,
        )
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_angle = axis * angle[..., None]
        rotations = R.from_rotvec(axis_angle).as_matrix()
        triangles = self.vertices[self.faces]
        triangles = np.einsum("abc,adc->abd", triangles, rotations)[..., :2]
        uv_matrices = np.stack(
            [triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]],
            axis=-1,
        )
        self.uv_matrices = np.linalg.inv(uv_matrices)
