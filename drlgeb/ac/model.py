import tensorflow as tf
from drlgeb.common import CnnEmbedding


class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_shape: tuple, action_size: int):
        super(ActorCriticModel, self).__init__()
        self.action_size = action_size
        self.embedding_layer = CnnEmbedding(state_shape)
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.values = tf.keras.layers.Dense(1)

    def call(self, inputs):
        embedding = self.embedding_layer(inputs)
        logits = self.policy_logits(embedding)
        values = self.values(embedding)
        return logits, values
