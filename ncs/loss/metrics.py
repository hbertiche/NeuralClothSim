import tensorflow as tf
from tensorflow.keras.metrics import Metric


class MyMetric(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name="m/" + name, **kwargs)
        self.error = self.add_weight(name="error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, error):
        self.error.assign_add(error)
        self.count.assign_add(1.0)

    def result(self):
        return self.error / self.count
