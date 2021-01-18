import abc
import datetime
import tensorflow as tf
import os


class Agent(object, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './train_logs/train-{}-{}/'.format(kwargs["name"], current_time)
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.train_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.mean_score = tf.keras.metrics.Mean('mean score', dtype=tf.float32)
        self.max_score = tf.keras.metrics.Mean('max score', dtype=tf.float32)

        checkpoint_path = "train-%s/cp-{epoch:04d}.ckpt" % (kwargs["name"])
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1))
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  './train_logs/train-{}-{}/'.format(kwargs["name"], current_time),
                                                  max_to_keep=5)

    @abc.abstractmethod
    def learn(self):
        pass

    def get_env(self):
        pass

    def train_summary(self, step, loss=None, mean_score=None, max_score=None):

        with self.summary_writer.as_default():
            if loss:
                self.train_loss(loss)
                tf.summary.scalar('loss', self.train_loss.result(), step=step)
                self.train_loss.reset_states()
            if mean_score:
                self.mean_score(mean_score)
                tf.summary.scalar('mean score', self.mean_score.result(), step=step)
                self.mean_score.reset_states()
            if max_score:
                self.max_score(max_score)
                tf.summary.scalar('max score', self.max_score.result(), step=step)
                self.max_score.reset_states()

    def checkpoint_save(self):
        self.manager.save()
