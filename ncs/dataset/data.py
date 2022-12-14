import os
import json
from random import shuffle
import numpy as np
from math import ceil

from tensorflow.keras.utils import Sequence

from global_vars import DATA_DIR, TXT_DIR, BODY_DIR
from dataset.sequence import PoseSequence


class Data(Sequence):
    def __init__(self, config, mode="train"):
        assert mode in {"train", "validation", "test"}, (
            "Data error: wrong mode. It is: "
            + mode
            + ". It must be in {'train', 'validation' ,'test'}"
        )
        self.config = config
        self.mode = mode
        self.txt = os.path.join(
            TXT_DIR, config.data.dataset, getattr(config.data, mode)
        )
        self.read_txt()
        self.read_sequences()
        self.batch_size = config.experiment.batch_size
        self.read_skeleton()
        self.make_reflection_map()
        self.reflect_probability = config.experiment.reflect_probability
        self.make_sample_list()

    # Read names of sequence files from a txt
    def read_txt(self):
        with open(self.txt, "r") as f:
            self.sequences = [
                os.path.join(DATA_DIR, self.config.data.dataset, line.replace("\n", ""))
                for line in f.readlines()
            ]

    # Loads the sequence data into PoseSequence objects (See 'sequence.py')
    def read_sequences(self):
        self.sequences = [PoseSequence(seq) for seq in self.sequences]
        self.seq_idx = np.array(range(self.num_sequences))
        self.seq_duration = np.array([seq.duration for seq in self.sequences])
        self.seq_prob = self.seq_duration / self.seq_duration.sum()

    # Reads skeleton metadata (joint names)
    def read_skeleton(self):
        fname = os.path.join(BODY_DIR, self.config.body.skeleton + "_skeleton.json")
        with open(fname, "r") as f:
            skel_data = json.load(f)
        self.joint_names = skel_data["joint_names"]
        self.num_joints = len(self.joint_names)

    # Mapping of joints for pose reflection augmentation
    def make_reflection_map(self):
        self.reflection_map = [None] * len(self.joint_names)
        for name, idx in self.joint_names.items():
            if "L" in name:
                name = name.replace("L", "R")
            elif "R" in name:
                name = name.replace("R", "L")
            self.reflection_map[idx] = self.joint_names[name]

    @property
    def num_sequences(self):
        return len(self.sequences)

    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def num_time_steps(self):
        return self.config.num_time_steps

    @property
    def skeleton_shape(self):
        return (self.num_joints, 4)

    @property
    def sample_poses_shape(self):
        return [self.num_time_steps, *self.skeleton_shape]

    @property
    def sample_trans_shape(self):
        return [self.num_time_steps, 3]

    # Makes array of consecutive time instants for the given experiment config
    def time_steps(self, time):
        if self.mode == "test":
            return np.arange(
                -2 * self.config.time_step, time + 1e-7, self.config.time_step
            )
        return time + np.arange(1 - self.num_time_steps, 1) * self.config.time_step

    # Reflects a pose (and translation) sequence
    def reflect(self, poses, trans):
        poses = poses[..., self.reflection_map, :]
        poses *= np.float32([1, 1, -1, -1])
        trans *= np.float32([-1, 1, 1])
        return poses, trans

    def __len__(self):
        if self.mode == "test":
            return len(self.sequences)
        return ceil(self.num_samples / self.batch_size)

    def __getitem__(self, idx):
        # Returns a single full sequence
        if self.mode == "test":
            seq = self.sequences[idx]
            t_seq = self.time_steps(seq.duration)
            return seq.get(t_seq, extrapolation="clip")
        # Returns a batch of subsequences with fixed length (temporal window size)
        start = idx * self.batch_size
        end = start + self.batch_size
        samples = self.samples[start:end]
        batch_size = len(samples)
        poses = np.zeros((batch_size, *self.sample_poses_shape), np.float32)
        trans = np.zeros((batch_size, *self.sample_trans_shape), np.float32)
        for i, (seq, t) in enumerate(samples):
            t_seq = self.time_steps(t)
            poses[i], trans[i] = self.sequences[seq].get(t_seq, extrapolation="clip")
            if self.mode == "train" and np.random.uniform() < self.reflect_probability:
                poses[i], trans[i] = self.reflect(poses[i], trans[i])
        return poses, trans

    # Make sample list with samples as (seq, t)
    def make_sample_list(self):
        idx, t = 0, 0
        self.samples = []
        for idx in self.seq_idx:
            t = np.arange(
                0,
                self.seq_duration[idx] + np.finfo(np.float32).eps,
                self.config.time_step,
            )
            self.samples += list(zip([idx] * len(t), t))
        if self.mode == "train":
            shuffle(self.samples)

    def on_epoch_end(self):
        if self.mode == "train":
            shuffle(self.samples)
