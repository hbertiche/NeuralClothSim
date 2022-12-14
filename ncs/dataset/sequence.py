import numpy as np

from utils.rotation import slerp, axis_angle_to_quat


class PoseSequence:
    def __init__(self, npz_file):
        with np.load(npz_file) as data:
            self.poses = data["poses"].astype(np.float32)
            self.trans = data["trans"].astype(np.float32)
            self.fps = data["fps"]
            self.source = data["source"]
        if self.poses.shape[-1] == 3:
            self.poses = axis_angle_to_quat(self.poses)

    @property
    def num_frames(self):
        return self.poses.shape[0]

    @property
    def duration(self):
        return (self.num_frames - 1) / self.fps

    @property
    def dt(self):
        return 1 / self.fps

    @property
    def num_joints(self):
        return self.poses.shape[1]

    @property
    def skeleton_shape(self):
        return self.poses.shape[1:]

    # Reads body pose and translation for the times specified in the nd-array 't'
    # Supports interpolation and extrapolation
    # Given an input 't' with dimensionality [t0, t1, ..., tn]
    # Returns pose as [t0, t1, ..., tn, num_joints, 4]
    # and translation as [t0, t1, ..., tn, 3]
    def get(self, t, extrapolation=None):
        batch_shape = t.shape
        t = t.reshape(-1)
        t = self.extrapolate(t, extrapolation)
        frame = t * self.fps
        frame, r = np.int32(frame), np.float32(frame % 1)
        next_frame = np.minimum(frame + 1, self.num_frames - 1)
        # Pose interpolation
        prev = self.poses[frame]
        next = self.poses[next_frame]
        pose = np.where(
            np.expand_dims(frame == next_frame, [-2, -1]), prev, slerp(prev, next, r)
        )
        # Translation interpolation
        prev = self.trans[frame]
        next = self.trans[next_frame]
        r = np.expand_dims(r, axis=-1)
        trans = (1 - r) * prev + r * next
        # Reshape into input batch shape
        pose = pose.reshape((*batch_shape, *self.skeleton_shape))
        trans = trans.reshape((*batch_shape, 3))
        return pose, trans

    # Extrapolates by mapping values of 't' outside sequence duration [0, T] to values within
    def extrapolate(self, t, mode=None):
        assert mode in {
            None,
            "clip",
            "mirror",
        }, "Wrong time extrapolation mode, must be in {None, 'clip', 'mirror'}"
        t = np.array(t)
        if (t >= 0.0).all() and (t <= self.duration).all():
            return t
        if mode is None:
            raise Exception("Queried time is outside the length of the sequence.")
        if mode == "clip":
            return np.clip(t, 0.0, self.duration)
        if mode == "mirror":
            t = np.where(t < 0, -t, t)
            t = np.where(t > self.duration, 2 * self.duration - t, t)
            return self.extrapolate(t, mode)
