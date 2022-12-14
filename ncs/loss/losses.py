import tensorflow as tf
from utils.mesh import vertex_normals, face_normals

# Mass-spring model
class EdgeLoss:
    def __init__(self, garment):
        self.edges = garment.edges
        self.edge_lengths_true = garment.edge_lengths

    @tf.function
    def __call__(self, vertices):
        edges = tf.gather(vertices, self.edges[:, 0], axis=1) - tf.gather(
            vertices, self.edges[:, 1], axis=1
        )
        edge_lengths = tf.norm(edges, axis=-1)
        edge_difference = edge_lengths - self.edge_lengths_true
        loss = edge_difference**2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.abs(edge_difference)
        error = tf.reduce_mean(error)
        return loss, error


# Baraff '98 cloth model (squared)
class ClothLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices

    @tf.function
    def __call__(self, vertices):
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=2,
        )
        w = tf.einsum("abcd,bce->abed", dX, self.uv_matrices)

        stretch = tf.norm(w, axis=-1) - 1
        stretch_loss = self.face_areas[:, None] * stretch**2
        stretch_loss = tf.reduce_sum(stretch_loss, axis=[1, 2])
        stretch_loss = tf.reduce_mean(stretch_loss)
        stretch_error = (
            self.face_areas[:, None] * tf.abs(stretch) * (0.5 / self.total_area)
        )
        stretch_error = tf.reduce_mean(tf.reduce_sum(stretch_error, axis=-1))

        shear = tf.reduce_sum(tf.multiply(w[:, :, 0], w[:, :, 1]), axis=-1)
        shear_loss = shear**2
        shear_loss *= self.face_areas
        shear_loss = tf.reduce_sum(shear_loss, axis=1)
        shear_loss = tf.reduce_mean(shear_loss)
        shear_error = self.face_areas * tf.abs(shear) * (1 / self.total_area)
        shear_error = tf.reduce_mean(tf.reduce_sum(shear_error, axis=-1))

        return stretch_loss, stretch_error, shear_loss, shear_error


# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, garment, l, m):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices
        self.l = l
        self.m = m

    @tf.function
    def __call__(self, vertices):
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=-1,
        )
        F = dX @ self.uv_matrices
        Ft = tf.linalg.matrix_transpose(F)
        G = 0.5 * (Ft @ F - tf.eye(2))
        S = self.m * G + (0.5 * self.l * tf.einsum("...ii", G))[
            ..., None, None
        ] * tf.eye(2, batch_shape=tf.shape(G)[:2])
        loss = tf.einsum("...ii", tf.linalg.matrix_transpose(S) @ G)
        loss *= self.face_areas
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = loss / (self.total_area)

        return loss, error


class BendingLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_adjacency = garment.face_adjacency
        face_areas = garment.face_areas[garment.face_adjacency].sum(-1)
        edge_lengths = garment.face_adjacency_edge_lengths
        self.stiffness_scaling = edge_lengths**2 / (8 * face_areas)
        self.angle_true = garment.face_dihedral

    @tf.function
    def __call__(self, vertices):
        mesh_face_normals = face_normals(vertices, self.faces)
        normals0 = tf.gather(mesh_face_normals, self.face_adjacency[:, 0], axis=1)
        normals1 = tf.gather(mesh_face_normals, self.face_adjacency[:, 1], axis=1)
        cos = tf.einsum("abc,abc->ab", normals0, normals1)
        sin = tf.norm(tf.linalg.cross(normals0, normals1), axis=-1)
        angle = tf.math.atan2(sin, cos) - self.angle_true
        loss = angle**2
        error = tf.abs(angle)
        loss *= self.stiffness_scaling
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)
        return loss, error


# Fast estimation of SDF
class CollisionLoss:
    def __init__(self, body, collision_threshold=0.004):
        self.body_faces = body.faces
        self.collision_vertices = tf.constant(body.collision_vertices)
        self.collision_threshold = collision_threshold

    @tf.function
    def __call__(self, vertices, body_vertices, indices):
        # Compute body vertex normals
        body_vertex_normals = vertex_normals(body_vertices, self.body_faces)
        # Gather collision vertices
        body_vertices = tf.gather(body_vertices, self.collision_vertices, axis=1)
        body_vertex_normals = tf.gather(
            body_vertex_normals, self.collision_vertices, axis=1
        )
        # Compute loss
        cloth_to_body = vertices - tf.gather_nd(body_vertices, indices)
        body_vertex_normals = tf.gather_nd(body_vertex_normals, indices)
        normal_dist = tf.einsum("abc,abc->ab", cloth_to_body, body_vertex_normals)
        loss = tf.minimum(normal_dist - self.collision_threshold, 0.0) ** 2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.math.less(normal_dist, 0.0)
        error = tf.cast(error, tf.float32)
        error = tf.reduce_mean(error)
        return loss, error


class GravityLoss:
    def __init__(self, vertex_area, density=0.15, gravity=[0, 0, -9.81]):
        self.vertex_mass = density * vertex_area[:, None]
        self.gravity = tf.constant(gravity, tf.float32)

    @tf.function
    def __call__(self, vertices):
        loss = -self.vertex_mass * vertices * self.gravity
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        return loss


class InertiaLoss:
    def __init__(self, dt, vertex_area, density=0.15):
        self.dt = dt
        self.vertex_mass = density * vertex_area
        self.total_mass = tf.reduce_sum(self.vertex_mass)

    @tf.function
    def __call__(self, vertices):
        x0, x1, x2 = tf.unstack(vertices, axis=1)
        x_proj = 2 * x1 - x0
        x_proj = tf.stop_gradient(x_proj)
        dx = x2 - x_proj
        loss = (0.5 / self.dt**2) * self.vertex_mass[:, None] * dx**2
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = self.vertex_mass * tf.norm(dx, axis=-1)
        error = tf.reduce_sum(error, axis=-1) / self.total_mass
        error = tf.reduce_mean(error)
        return loss, error


class PinningLoss:
    def __init__(self, garment, pin_blend_weights=False):
        self.indices = garment.pinning_vertices
        self.vertices = garment.vertices[self.indices]
        self.pin_blend_weights = pin_blend_weights
        if pin_blend_weights:
            self.blend_weights = garment.blend_weights[self.indices]

    @tf.function
    def __call__(self, unskinned, blend_weights):
        loss = tf.gather(unskinned, self.indices, axis=-2) - self.vertices
        loss = loss**2
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        if self.pin_blend_weights:
            _loss = tf.gather(blend_weights, self.indices, axis=-2) - self.blend_weights
            _loss = _loss**2
            _loss = tf.reduce_mean(_loss, 0)
            loss += 1e2 * tf.reduce_sum(_loss)
        return loss
