import os
import json
from math import ceil

from global_vars import NUM_JOINTS, GRAVITY


class Config:
    def __init__(self, path_or_dict):
        if isinstance(path_or_dict, str):
            assert os.path.isfile(
                path_or_dict
            ), "Config class error: JSON config file does not exist."
            self.name = os.path.basename(os.path.splitext(path_or_dict)[0])
            with open(path_or_dict, "r") as f:
                self.json_dict = json.load(f)
        else:
            self.json_dict = path_or_dict

        for name, value in self.json_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, name, value)

    def to_json(self):
        return json.dumps(self.json_dict)

    def __str__(self):
        return json.dumps(self.json_dict, indent=4)


class MainConfig(Config):
    @property
    def learning_rate(self):
        return self.experiment.learning_rate

    @property
    def time_step(self):
        return 1 / self.data.fps

    @property
    def num_time_steps(self):
        return ceil(self.experiment.temporal_window_size / self.time_step) + 3

    @property
    def blend_weights_trainable(self):
        return self.model.blend_weights_optimize

    @property
    def pin_blend_weights(self):
        return self.blend_weights_trainable

    @property
    def cloth_model(self):
        return self.loss.cloth.type

    @property
    def gravity(self):
        g = self.loss.gravity
        if not isinstance(g, str):
            return g
        sign, axis = g[0], g[1]
        sign = 1 if sign == "+" else -1
        if axis == "X":
            return [sign * GRAVITY, 0, 0]
        if axis == "Y":
            return [0, sign * GRAVITY, 0]
        if axis == "Z":
            return [0, 0, sign * GRAVITY]

    @property
    def motion_augmentation(self):
        return int(self.experiment.batch_size * self.experiment.motion_augmentation)

    @property
    def input_shape(self):
        num_joints = NUM_JOINTS[self.body.skeleton]
        shape = [
            (None, self.num_time_steps, num_joints, 4),
            (None, self.num_time_steps, 3),
        ]
        return shape
