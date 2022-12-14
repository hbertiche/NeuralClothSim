import os
from utils.tensor import compute_nth_derivative
import tensorflow as tf
from tensorflow.keras.layers import GRU
from loss.losses import *
from loss.metrics import *
from model.body import Body

from model.cloth import Garment
from .layers import *
from global_vars import BODY_DIR


class NCS(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        folder = os.path.join(BODY_DIR, config.body.model)
        body_model = os.path.join(folder, "body.npz")
        garment_obj = os.path.join(folder, config.garment.name)
        # Read body
        print("Reading body model...")
        self.body = Body(body_model, input_joints=config.body.input_joints)
        # Read garment
        print("Reading garment...")
        self.garment = Garment(garment_obj)
        print("Computing cloth blend weights...")
        self.garment.transfer_blend_weights(self.body)
        print("Smoothing cloth blend weights...")
        self.garment.smooth_blend_weights(
            iterations=config.garment.blend_weights_smoothing_iterations
        )

        # Build model
        self.build_model()

        # Losses/Metrics
        self.build_losses_and_metrics()

    def build_model(self):
        self.build_lbs()
        self.build_encoder()
        self.build_decoder()

    def build_lbs(self):
        self.rot = Rotation(name="Rotation")
        self.skeleton = Skeleton(self.body.joints, name="Skeleton")
        self.lbs_body = LBS(self.body.blend_weights, trainable=False, name="LBS/Body")
        self.lbs_cloth = LBS(
            self.garment.blend_weights,
            trainable=self.config.blend_weights_trainable,
            name="LBS/Cloth",
        )

    def build_encoder(self):
        self.static_encoder = [
            SkelFlatten(),
            FullyConnected(64, act=tf.nn.relu, name="stc_enc/fc0"),
            FullyConnected(128, act=tf.nn.relu, name="stc_enc/fc0"),
            FullyConnected(256, act=tf.nn.relu, name="stc_enc/fc0"),
            FullyConnected(512, act=tf.nn.relu, name="stc_enc/fc0"),
        ]
        self.dynamic_encoder = [
            FullyConnected(32, act=tf.nn.relu, use_bias=False, name="dyn_enc/fc0"),
            FullyConnected(32, act=tf.nn.relu, use_bias=False, name="dyn_enc/fc1"),
            SkelFlatten(),
            FullyConnected(512, act=tf.nn.relu, use_bias=False, name="dyn_enc/fc2"),
            FullyConnected(512, act=tf.nn.relu, use_bias=False, name="dyn_enc/fc3"),
            GRU(512, use_bias=False, return_sequences=True, name="dyn_enc/gru"),
        ]

    def build_decoder(self):
        self.decoder = [
            FullyConnected(512, act=tf.nn.relu, name="dec/fc0"),
            FullyConnected(512, act=tf.nn.relu, name="dec/fc1"),
            FullyConnected(512, act=tf.nn.relu, name="dec/fc2"),
            PSD(self.garment.num_verts, name="dec/PSD"),
        ]

    def build_losses_and_metrics(self):
        # Losses and Metrics
        self.loss_metric = MyMetric(name="Loss")
        # Cloth model
        if self.config.cloth_model == "mass-spring":
            self.cloth_loss = EdgeLoss(self.garment)
            self.edge_metric = MyMetric(name="Edge")
        elif self.config.cloth_model == "baraff98":
            self.cloth_loss = ClothLoss(self.garment)
            self.stretch_metric = MyMetric(name="Stretch")
            self.shear_metric = MyMetric(name="Shear")
        elif self.config.cloth_model == "stvk":
            self.cloth_loss = StVKLoss(
                self.garment,
                self.config.loss.cloth.lambda_,
                self.config.loss.cloth.mu,
            )
            self.strain_metric = MyMetric(name="Strain")
        # Bending
        self.bending_loss = BendingLoss(self.garment)
        self.bending_metric = MyMetric(name="Bending")
        # Collision
        self.collision = Collision(self.body, use_ray=False, name="Collision")
        self.collision_loss = CollisionLoss(
            self.body, collision_threshold=self.config.loss.collision_threshold
        )
        self.collision_metric = MyMetric(name="Collision")
        # Gravity
        self.gravity_loss = GravityLoss(
            self.garment.vertex_area,
            density=self.config.loss.density,
            gravity=self.config.gravity,
        )
        self.gravity_metric = MyMetric(name="Gravity")
        # Intertia
        self.inertia_loss = InertiaLoss(
            self.config.time_step,
            self.garment.vertex_area,
            density=self.config.loss.density,
        )
        self.inertia_metric = MyMetric(name="Inertia")
        # Pinning (if)
        if self.garment.pinning:
            self.pinning_loss = PinningLoss(self.garment, self.config.pin_blend_weights)

    def compute_losses_and_metrics(self, body, vertices, unskinned, training):
        loss = self.compute_static_losses_and_metrics(body, vertices[:, -1], unskinned)
        if training and self.config.motion_augmentation:
            vertices = vertices[self.config.motion_augmentation :]
        loss += self.compute_dynamic_losses_and_metrics(vertices)
        return loss

    def compute_static_losses_and_metrics(self, body, vertices, unskinned):
        # Cloth
        if self.config.cloth_model == "mass-spring":
            cloth_loss, edge_error = self.cloth_loss(vertices)
            cloth_loss *= self.config.loss.cloth.edge
        elif self.config.cloth_model == "baraff98":
            stretch_loss, stretch_error, shear_loss, shear_error = self.cloth_loss(
                vertices
            )
            cloth_loss = (
                self.config.loss.cloth.stretch * stretch_loss
                + self.config.loss.cloth.shear * shear_loss
            )
        elif self.config.cloth_model == "stvk":
            cloth_loss, strain_error = self.cloth_loss(vertices)
        # Bending
        bending_loss, bending_error = self.bending_loss(vertices)
        # Collision
        collision_indices = self.collision(vertices, body)
        collision_loss, collision_error = self.collision_loss(
            vertices, body, collision_indices
        )
        # Gravity
        gravitational_potential = self.gravity_loss(vertices)
        # Pinning
        if self.garment.pinning:
            pinning_loss = self.pinning_loss(unskinned, self.cloth_blend_weights)
        # Combine loss
        loss = (
            cloth_loss
            + self.config.loss.bending * bending_loss
            + self.config.loss.collision_weight * collision_loss
            + gravitational_potential
        )
        if self.garment.pinning:
            loss += self.config.loss.pinning * pinning_loss
        # Update metrics
        self.loss_metric.update_state(loss)
        if self.config.cloth_model == "mass-spring":
            self.edge_metric.update_state(edge_error)
        elif self.config.cloth_model == "baraff98":
            self.stretch_metric.update_state(stretch_error)
            self.shear_metric.update_state(shear_error)
        elif self.config.cloth_model == "stvk":
            self.strain_metric.update_state(strain_error)
        self.bending_metric.update_state(bending_error)
        self.collision_metric.update_state(collision_error)
        self.gravity_metric.update_state(gravitational_potential)
        return loss

    def compute_dynamic_losses_and_metrics(self, vertices):
        inertia_loss, inertia_error = self.inertia_loss(vertices)
        self.inertia_metric.update_state(inertia_error)
        return inertia_loss

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            body, vertices, unskinned = self(inputs, training=True)
            loss = self.compute_losses_and_metrics(
                body, vertices, unskinned, training=True
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        body, vertices, unskinned = self(inputs, training=False)
        self.compute_losses_and_metrics(body, vertices, unskinned, training=False)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        if self.config.cloth_model == "mass-spring":
            cloth_metrics = [self.edge_metric]
        elif self.config.cloth_model == "baraff98":
            cloth_metrics = [self.stretch_metric, self.shear_metric]
        elif self.config.cloth_model == "stvk":
            cloth_metrics = [self.strain_metric]
        return [
            self.loss_metric,
            *cloth_metrics,
            self.bending_metric,
            self.collision_metric,
            self.gravity_metric,
            self.inertia_metric,
        ]

    @property
    def cloth_blend_weights(self):
        return self.lbs_cloth.blend_weights

    def predict(self, inputs, w):
        poses, trans = inputs
        assert (
            poses.ndim == 3
        ), "Pose sequence has wrong dimensions. Should be (T, J, 3/4)."
        poses, trans = poses[None], trans[None]
        X, matrices = self.call_inputs(poses, trans)
        deformations = self.call_network(X, w=w, training=False, predict=True)
        body = self.lbs_body(self.body.vertices, matrices)
        unskinned = self.garment.vertices + deformations
        matrices = tf.gather(matrices, self.body.input_joints, axis=-3)
        garment = self.lbs_cloth(unskinned, matrices)
        return body[0], garment[0], unskinned[0]

    def call(self, inputs, w=None, training=False):
        poses, trans = inputs
        # Make input feats
        X, matrices = self.call_inputs(poses, trans)
        # Call network
        deformations = self.call_network(X, w=w, training=training)
        # Compute body LBS
        body = self.lbs_body(self.body.vertices, matrices[:, -1])
        # Compute garment LBS
        unskinned = self.garment.vertices + deformations
        matrices = tf.gather(matrices, self.body.input_joints, axis=-3)
        garment = self.lbs_cloth(unskinned, matrices[:, -3:])
        return body, garment, unskinned[:, -1]

    def call_inputs(self, poses, trans):
        # Compute local rotation matrices
        rotations = self.rot(poses)
        # Compute global transformation matrices
        matrices = self.body.forward_kinematics(rotations, trans)
        matrices_inv = tf.linalg.matrix_transpose(matrices[..., :3])
        # 6D descriptors
        X = tf.reshape(rotations[..., :2], (*tf_shape(rotations)[:-2], 6))
        # Unposed gravity
        Z = (
            matrices_inv
            @ self.gravity_loss.gravity[:, None]
            * (1 / tf.norm(self.gravity_loss.gravity))
        )[..., 0]
        # Combine
        X = tf.concat((X, Z), axis=-1)
        # Compute joint locations
        J = self.skeleton(matrices)
        # Compute joint temporal derivatives
        dX = compute_nth_derivative(X, 1, self.config.time_step)[:, 1:]
        dJ = compute_nth_derivative(J, 2, self.config.time_step)
        # Unpose accelerations
        dJ = (matrices_inv[:, 2:] @ dJ[..., None])[..., 0]
        # Combine
        X = tf.concat((X[:, 2:], dX, dJ), axis=-1)
        # Gather relevant joints only
        X = tf.gather(X, self.body.input_joints, axis=-2)
        return X, matrices[:, 2:]

    def call_network(self, x, w, training, predict=False):
        # Split static and dynamic features
        x_static, x_dynamic = tf.split(x, [9, 12], axis=-1)
        # Static encoder
        if not predict:
            x_static = x_static[:, -3:]
        for l in self.static_encoder:
            x_static = l(x_static)
        # Dynamic encoder
        for l in self.dynamic_encoder:
            x_dynamic = l(x_dynamic)
        if not predict:
            x_dynamic = x_dynamic[:, -3:]
        if training and self.config.motion_augmentation:
            x_static, x_dynamic = self.motion_augmentation(x_static, x_dynamic)
        if w is not None:
            x_dynamic = w * x_dynamic
        x = x_static + x_dynamic

        for l in self.decoder:
            x = l(x)

        return x

    def motion_augmentation(self, x_static, x_dynamic):
        batch_size = tf.shape(x_static)[0]
        n = self.config.motion_augmentation
        splits = [n, tf.maximum(0, batch_size - n)]
        x_static_aug, x_static = tf.split(x_static, splits)
        x_dynamic_aug, x_dynamic = tf.split(x_dynamic, splits)
        x_static_aug = tf.stop_gradient(x_static_aug)
        x_dynamic_aug = tf.random.shuffle(x_dynamic_aug)
        x_dynamic_aug = tf.stop_gradient(x_dynamic_aug)
        x_static = tf.concat((x_static_aug, x_static), axis=0)
        x_dynamic = tf.concat((x_dynamic_aug, x_dynamic), axis=0)
        return x_static, x_dynamic
