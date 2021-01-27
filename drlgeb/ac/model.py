import tensorflow as tf
from drlgeb.common import CnnEmbedding, DenseEmbedding


class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_shape, action_size):
        super(ActorCriticModel, self).__init__()
        self.action_size = action_size
        if len(state_shape) == 1:
            self.embedding_layer = DenseEmbedding()
        else:
            self.embedding_layer = CnnEmbedding(state_shape)
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.values = tf.keras.layers.Dense(1)

    def call(self, inputs):
        embedding = self.embedding_layer(inputs)
        logits = self.policy_logits(embedding)
        values = self.values(embedding)
        return logits, values
