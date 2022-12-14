import os
import sys
import argparse
from shutil import rmtree
import tensorflow as tf

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig
from global_vars import LOGS_DIR, CHECKPOINTS_DIR


def make_model(config):
    model = NCS(config)
    print("Building model...")
    model.build(input_shape=config.input_shape)
    model.summary()
    print("Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer)
    if config.experiment.checkpoint is not None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, config.experiment.checkpoint)
        model.load_weights(checkpoint_path)
    return model


def main(config):
    # Remove previous runs logs and checkpoints for this experiment
    log_dir = os.path.join(LOGS_DIR, config.name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, config.name)
    if os.path.isdir(log_dir):
        rmtree(log_dir)
    if os.path.isdir(checkpoint_dir):
        rmtree(checkpoint_dir)

    print("Initializing model...")
    if len(gpus) > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = make_model(config)
    else:
        model = make_model(config)

    print("Reading data...")
    data = Data(config, mode="train")
    validation_data = Data(config, mode="validation")

    print("Training...")
    model.fit(
        data,
        validation_data=validation_data,
        epochs=config.experiment.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                write_graph=False,
                write_steps_per_second=False,
                update_freq="epoch",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir, save_freq="epoch"
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    opts = parser.parse_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

    # Limit VRAM usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")
        sys.exit()

    config = MainConfig(opts.config)
    main(config)
