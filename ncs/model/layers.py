import tensorflow as tf

from tensorflow.keras.layers import Layer
from scipy.spatial import cKDTree
import ray

from utils.rotation import from_axis_angle, from_quaternion
from utils.mesh import lbs
from utils.tensor import tf_shape


class FullyConnected(Layer):
    def __init__(self, units, act=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.act = act or (lambda x: x)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="bias"
            )

    def call(self, x):
        x = tf.expand_dims(x, axis=-2)
        x = x @ self.kernel
        x = x[..., 0, :]
        if self.use_bias:
            x += self.bias
        x = self.act(x)
        return x


class SkelFlatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def call(self, inputs):
        input_shape = tf_shape(inputs)
        return tf.reshape(inputs, (*input_shape[:-2], tf.reduce_prod(input_shape[-2:])))


class PSD(Layer):
    def __init__(self, num_verts, num_dims=3, act=None, **kwargs):
        super().__init__(**kwargs)
        self.num_verts = num_verts
        self.num_dims = num_dims
        self.act = act or (lambda x: x)

    def build(self, input_shape):
        shape = input_shape[-1], self.num_verts, self.num_dims
        self.psd = tf.Variable(tf.initializers.glorot_normal()(shape), name="psd")

    def call(self, x):
        x = tf.tensordot(x, self.psd, axes=[[-1], [0]])
        x = self.act(x)
        return x


class Skeleton(Layer):
    def __init__(self, rest_joints, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.rest_joints = rest_joints

    def call(self, matrices):
        return lbs(self.rest_joints, matrices)


class LBS(Layer):
    def __init__(self, blend_weights, trainable, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        if trainable:
            blend_weights = tf.math.log(blend_weights + 0.001)
        self.blend_weights = self.add_weight(
            shape=blend_weights.shape,
            initializer=tf.keras.initializers.Constant(blend_weights),
            trainable=trainable,
            name="blend_weights",
        )

    def call(self, vertices, matrices):
        if self.trainable:
            blend_weights = tf.nn.softmax(self.blend_weights)
            return lbs(vertices, matrices, blend_weights)
        return lbs(vertices, matrices, self.blend_weights)


class Rotation(Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def build(self, input_shape):
        if input_shape[-1] == 3:
            self.mode = "axis_angle"
        elif input_shape[-1] == 4:
            self.mode = "quaternion"

    def call(self, orientations):
        if self.mode == "axis_angle":
            return from_axis_angle(orientations)
        elif self.mode == "quaternion":
            return from_quaternion(orientations)


class Collision(Layer):
    def __init__(self, body, use_ray, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)
        self.collision_vertices = tf.constant(body.collision_vertices)
        self.use_ray = use_ray
        self.run_sample = lambda elem: cKDTree(elem[1]).query(elem[0], workers=-1)[1]
        if use_ray:
            ray.init()
            self.run_sample = ray.remote(self.run_sample)

    def run(self, vertices, collider):
        if self.use_ray:
            return ray.get(
                [self.run_sample.remote(elem) for elem in zip(vertices, collider)]
            )
        return tf.stack([self.run_sample(elem) for elem in zip(vertices, collider)])

    def build(self, input_shape):
        batch_size, num_verts = input_shape[:2]
        self.idx = tf.tile(tf.range(batch_size)[:, None], [1, num_verts])

    def call(self, vertices, collider):
        batch_size = tf.shape(vertices)[0]
        idx = self.run(vertices, tf.gather(collider, self.collision_vertices, axis=-2))
        idx = tf.stack(
            [tf.gather(self.idx, tf.range(batch_size), axis=0), idx], axis=-1
        )
        return idx
