import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from utils.tensor import tf_shape


def triangulate(faces): #三角化
    triangles = np.int32(
        [triangle for polygon in faces for triangle in _triangulate_recursive(polygon)]# 提取faces三角化为列表
    )
    ''' 
    # 等效于下面的代码
        triangles = []
        for polygon in faces:
            for triangle in _triangulate_recursive(polygon):
                triangles.append(triangle)
    '''
    return triangles


def _triangulate_recursive(face):
    if len(face) == 3:
        return [face]
    else:
        return [face[:3]] + _triangulate_recursive([face[0], *face[2:]]) # *表示解包元素并合并# 
                                                                         # 等效于facep[0]+face[2:]
        #第一个顶点与其余顶点中两个顶点组合为三角形

def faces_to_edges_and_adjacency(faces):
    edges = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    for edge, face_list in edges.items():
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_edges += [edge]
    edges = np.array([list(edge) for edge in edges.keys()], np.int32)
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    return edges, face_adjacency, face_adjacency_edges


def laplacian_matrix(faces):
    G = {}
    for face in faces:
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            if v not in G:
                G[v] = {}
            if nv not in G:
                G[nv] = {}
            G[v][nv] = 1
            G[nv][v] = 1
    return graph_laplacian(G)


def graph_laplacian(graph):
    row, col, data = [], [], []
    for v in graph:
        n = len(graph[v])
        row += [v] * n
        col += [u for u in graph[v]]
        data += [1.0 / n] * n
    return coo_matrix((data, (row, col)), shape=[len(graph)] * 2)


@tf.function
def face_normals(vertices, faces, normalized=True):
    input_shape = vertices.get_shape()
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    v01 = tf.gather(vertices, faces[:, 1], axis=1) - tf.gather(
        vertices, faces[:, 0], axis=1
    )
    v12 = tf.gather(vertices, faces[:, 2], axis=1) - tf.gather(
        vertices, faces[:, 1], axis=1
    )
    normals = tf.linalg.cross(v01, v12)
    if normalized:
        normals /= tf.norm(normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    normals = tf.reshape(normals, (*input_shape[:-2], -1, 3))
    return normals


@tf.function
def vertex_normals(vertices, faces):
    input_shape = vertices.get_shape()
    batch_size = tf.reduce_prod(input_shape[:-2] or [1])
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    # Compute face normals
    mesh_normals = face_normals(vertices, faces, normalized=False)
    # Scatter face normals
    faces_batched = tf.stack(
        (
            tf.tile(tf.range(batch_size)[:, None, None], [1, *faces.get_shape()]),
            tf.tile(faces[None], [batch_size, 1, 1]),
        ),
        axis=-1,
    )
    mesh_normals = tf.tile(mesh_normals[:, :, None], [1, 1, 3, 1])
    vertex_normals = tf.zeros((batch_size, *input_shape[-2:]), tf.float32)
    vertex_normals = tf.tensor_scatter_nd_add(
        vertex_normals, faces_batched, mesh_normals
    )
    vertex_normals /= (
        tf.norm(vertex_normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    )
    # Reshape back to input shape
    vertex_normals = tf.reshape(vertex_normals, input_shape)
    return vertex_normals


def edge_lengths(vertices, edges):
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = np.einsum("ab,ab->a", normals0, normals1)
    sin = np.linalg.norm(np.cross(normals0, normals1), axis=-1)
    return np.arctan2(sin, cos)


def vertex_area(vertices, faces):
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    face_areas = np.linalg.norm(np.cross(v01, v12), axis=-1)
    vertex_areas = np.zeros((vertices.shape[0],), np.float32)
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i]
    vertex_areas *= 1 / 6
    total_area = vertex_areas.sum()
    return vertex_areas, face_areas, total_area


@tf.function
def lbs(vertices, matrices, blend_weights=None):
    matrices = tf.reshape(matrices, (*tf_shape(matrices)[:-2], 3 * 4))
    if blend_weights is not None:
        matrices = blend_weights @ matrices
    matrices = tf.reshape(matrices, (*tf_shape(matrices)[:-1], 3, 4))
    rotations, translations = tf.split(matrices, [3, 1], axis=-1)
    vertices = rotations @ vertices[..., None]
    vertices += translations
    return vertices[..., 0]
