import abc
import datetime
import tensorflow as tf
from drlgeb.common.logging_util import default_logger as logging
import os
from gym import wrappers
import time

class Agent(object, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = './train_logs/train-{}-{}/'.format(kwargs["name"], current_time)
        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=kwargs['opt'], net=kwargs['net'])
        # self.manager = tf.train.CheckpointManager(self.ckpt, self.train_log_dir, max_to_keep=5)
        assert getattr(self, 'model', None) is not None
        assert getattr(self, 'env', None) is not None
        assert getattr(self, 'state_shape', None) is not None
        assert getattr(self, 'action_size', None) is not None

        if kwargs.get('init_ckp_path', None):
            latest = tf.train.latest_checkpoint(kwargs.get('init_ckp_path'))
            self.model.load_weights(latest)
            # self.train_log_dir = kwargs.get('init_ckp_path')

        self.model.build((None,) + self.state_shape)
        self.model.summary()

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def record(self):
        pass

    @abc.abstractmethod
    def get_action(self, state):
        pass

    def play(self, episodes: int, model_path: str):
        self.env = wrappers.Monitor(self.env, './videos/' + str(time.time()) + '/')
        try:
            latest = tf.train.latest_checkpoint(model_path)
            print(latest)
            self.model.load_weights(latest)
        except:
            self.model = tf.keras.models.load_model(model_path)
        obs = self.env.reset()
        score = 0
        episode = 0
        while True:
            action = self.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            score += reward
            self.env.render()
            # time.sleep(0.03)
            if done:
                episode += 1
                print(f"Episode {episode} score:", score)
                if episode == episodes:
                    break
                obs = self.env.reset()
                score = 0
        self.env.close()

    def train_summary(self, step, **kwargs):
        if getattr(self, 'summary_writer', None) is None:
            self.__setattr__('summary_writer', tf.summary.create_file_writer(self.train_log_dir))
        with getattr(self, 'summary_writer').as_default():
            for k, v in kwargs.items():
                if getattr(self, k, None) is None:
                    self.__setattr__(k, tf.keras.metrics.Mean(k, dtype=tf.float32))

                getattr(self, k)(v)
                tf.summary.scalar(k, getattr(self, k).result(), step=step)
                getattr(self, k).reset_states()

    def checkpoint_save(self, num):
        self.model.save_weights(os.path.join(self.train_log_dir, str(num) + "-ckp"))
        logging.info("Save checkpoint successfully, in:{}".format(self.train_log_dir))
