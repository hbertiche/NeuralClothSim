import numpy as np
import tensorflow as tf

from utils.tensor import tf_shape


@tf.function
def from_axis_angle(axis_angle):
    input_shape = tf_shape(axis_angle)
    assert (
        input_shape[-1] == 3
    ), "Rotation from axis angle error. Wrong tensor size: " + str(input_shape)
    ndims = len(axis_angle.shape)

    # Flatten batch dimensions
    axis_angle = tf.reshape(axis_angle, (-1, 3))
    # Decompose into axis and angle
    angle = tf.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / (angle + tf.keras.backend.epsilon())

    # Compute rodrigues
    batch_size = tf.shape(axis_angle)[0]
    zeros = tf.zeros((batch_size,), tf.float32)
    M = tf.stack(
        [
            zeros,
            -axis[:, 2],
            axis[:, 1],
            axis[:, 2],
            zeros,
            -axis[:, 0],
            -axis[:, 1],
            axis[:, 0],
            zeros,
        ],
        axis=1,
    )
    M = tf.reshape(M, (-1, 3, 3))

    rotations = (
        tf.eye(3, batch_shape=[batch_size])
        + tf.sin(angle)[:, None] * M
        + (1 - tf.cos(angle)[:, None]) * (M @ M)
    )
    # Reshape back to original batch shape
    rotations = tf.reshape(rotations, (*input_shape[: ndims - 1], 3, 3))
    return rotations


@tf.function
def from_quaternion(quaternions):
    input_shape = tf_shape(quaternions)
    assert (
        input_shape[-1] == 4
    ), "Rotation from quaternion error. Wrong tensor size: " + str(input_shape)
    ndims = len(quaternions.shape)

    # Flatten batch dimensions
    quaternions = tf.reshape(quaternions, (-1, 4))

    # Compute rotations
    w, x, y, z = tf.unstack(quaternions, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    rotations = tf.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        axis=-1,
    )

    # Reshape back to original batch shape
    rotations = tf.reshape(rotations, (*input_shape[: ndims - 1], 3, 3))
    return rotations


def lerp(q0, q1, r):
    r = np.expand_dims(r, [-2, -1])
    return (1 - r) * q0 + r * q1


def slerp(q0, q1, r):
    r = np.expand_dims(r, axis=[-2, -1])
    dot = (q0 * q1).sum(-1, keepdims=True)
    dot = np.clip(dot, -1, 1)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    w0 = np.where(sin_omega, np.sin((1 - r) * omega) / sin_omega, 1 - r)
    w1 = np.where(sin_omega, np.sin(r * omega) / sin_omega, r)
    return w0 * q0 + w1 * q1


@tf.function
def tf_slerp(q0, q1, r):
    print("")
    print("DO NOT USE THIS FUNCTION")
    print(
        "For some reason, implementing the EXACT SAME function in TensorFlow is no stable"
    )
    print("")
    omega = tf.math.acos(tf.reduce_sum(q0 * q1, axis=-1))[..., None]
    sin_omega = tf.math.sin(omega) + tf.keras.backend.epsilon()
    return (tf.math.sin((1 - r) * omega) * q0 + tf.math.sin(r * omega) * q1) / sin_omega


def axis_angle_to_quat(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1)[..., None] + np.finfo(float).eps
    axis = rotvec / angle
    sin = np.sin(angle / 2)
    w = np.cos(angle / 2)
    return np.concatenate((w, sin * axis), axis=-1)


def quat_to_axis_angle(quat):
    angle = 2 * np.arccos(quat[..., 0:1])
    axis = quat[..., 1:] * (1 / (np.sin(angle / 2) + np.finfo(float).eps))
    return angle * axis
