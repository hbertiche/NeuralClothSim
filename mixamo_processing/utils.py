import numpy as np

def merge_meshes(V, F):
    # Merges multiple meshes into a single one
    # V : list of vertex arrays ([N x 3])
    # F : list of list of faces (works for n-gons)
    assert len(V) == len(F), "Number of vertex matrices and face lists are not consistent."
    vertices, faces = V[0], F[0]
    for i in range(1, len(V)):
        offset = vertices.shape[0]
        vertices = np.concatenate((vertices, V[i]), axis=0)
        F[i] = [[v_idx + offset for v_idx in f] for f in F[i]]
        faces += F[i]
    return vertices, faces


def triangulate(faces):
    # Naively triangulates faces of a mesh (does not consider geometry)
    def rec(face):
        if len(face) == 3:
            return [face]
        return [face[:3]] + rec([face[0], *face[2:]])

    triangles = np.int32([triangle for polygon in faces for triangle in rec(polygon)])
    return triangles