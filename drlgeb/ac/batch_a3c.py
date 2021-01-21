import sys

sys.path.append("../../")

from drlgeb.ac.model import ActorCriticModel
from drlgeb.common import make_atari, Agent
from collections import deque
import queue
import threading
import gym
from multiprocessing import Process, Pipe
import multiprocessing as mp
import tensorflow as tf
import numpy as np
from drlgeb.common.logging_util import default_logger as logging
import time

class RateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_rate, l: list):
        super(RateSchedule, self).__init__()
        self.init_rate = init_rate
        self.l = l

    def __call__(self, step):
        for i in range(len(self.l)):
            if step < self.l[i][0]:
                if i == 0:
                    return self.init_rate
                return self.l[i - 1][1]
        return self.l[-1][1]


class DenseAC(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DenseAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')

        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.values = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        v1 = self.dense2(x)
        logits = self.policy_logits(x)
        values = self.values(v1)
        return logits, values


class TransitionExperience(object):

    def __init__(self, state, action, reward, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class Master(Agent):
    class WorkerState(object):
        def __init__(self):
            self.memory = []
            self.score = 0

    def __init__(self, env_id="SpaceInvaders-v0", **configs):
        self.env_id = env_id
        self.env = make_atari(env_id=env_id)
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.model = ActorCriticModel(self.state_shape, self.action_size)
        self.nenvs = configs.get('nenvs', mp.cpu_count() * 2)
        self.lr = RateSchedule(configs.get('lr', 0.01), [(120000, 0.0003), (720000, 0.0001)])
        self.opt = tf.keras.optimizers.Adam(self.lr, epsilon=1e-3)
        self.local_time_max = configs.get('local_time_max', 5)
        self.gamma = configs.get('discount_gamma', 0.99)
        self.batch_size = configs.get('batch_size', 128)
        self.step_max = configs.get('step_max', 1e9)
        self.scores = deque(maxlen=100)

        super().__init__(name=env_id, **configs)

    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        logits, _ = self.model(state)
        policy = tf.nn.softmax(logits).numpy()[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def test_env(self, vis=False):
        state = self.env.reset()
        done = False
        score = 0
        while not done:
            next_state, reward, done, _ = self.env.step(self.get_action(state))
            state = next_state
            if vis:
                self.env.render()
            score += reward
        return score

    def update(self):
        step = 0
        while step < self.step_max:
            states = []
            actions = []
            discount_returns = []
            action_probs = []
            while True:
                state, action, R, action_prob = self.queue.get()
                states.append(state)
                actions.append(action)
                discount_returns.append(R)
                action_probs.append(action_prob)
                if len(states) == self.batch_size:
                    with tf.GradientTape() as tape:
                        states = np.array(states, dtype=np.float32)
                        actions = np.array(actions, dtype=np.int32)
                        discount_returns = np.array(discount_returns, dtype=np.float32)
                        action_probs = np.array(action_probs, dtype=np.float32)
                        logits, values = self.model(states)
                        values = tf.squeeze(values, [1])
                        policy = tf.nn.softmax(logits)
                        log_probs = tf.math.log(policy + 1e-6)
                        log_pi_a_given_s = tf.reduce_sum(log_probs * tf.one_hot(actions, self.action_size), 1)
                        advantage = tf.subtract(tf.stop_gradient(values), discount_returns)
                        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(actions, self.action_size), 1)
                        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_probs + 1e-8), 0, 10))
                        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance)
                        entropy_loss = tf.reduce_sum(policy * log_probs)
                        value_loss = tf.nn.l2_loss(values - discount_returns)
                        pred_reward = tf.reduce_mean(values)
                        loss = tf.add_n([policy_loss, entropy_loss * (0.01 if step < 480000 else 0.005),
                                         value_loss * 0.5]) / self.batch_size

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    # grads = [(tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))) for grad in grads]
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    step += 1
                    self.record(step=step, pred_reward=pred_reward, loss=loss, policy_loss=policy_loss,
                                entropy_loss=entropy_loss, value_loss=value_loss, importance=tf.reduce_mean(importance))
                    break

    def learn(self):

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.work_states = [self.WorkerState() for _ in range(self.nenvs)]
        self.ps = [Workers(i, remote, work_remote, self.env_id) for i, (remote, work_remote) in
                   enumerate(zip(self.remotes, self.work_remotes))]

        self.queue = queue.Queue(maxsize=self.batch_size * 2 * 8)

        for worker in self.ps:
            worker.start()
            print(f"{worker.name} Start!")

        t = threading.Thread(target=self.recv_send)
        t.start()

        update = threading.Thread(target=self.update)
        update.start()

        for worker in self.ps:
            worker.join()
        t.join()
        update.join()

    def recv_send(self):
        candidate = list(range(self.nenvs))
        while True:
            idxs = np.random.choice(candidate, 32)
            for idx in idxs:
                work_idx, state, reward, done = self.remotes[idx].recv()
                self.work_states[idx].score += reward
                if done:
                    self.scores.append(self.work_states[idx].score)
                    self.work_states[idx].score = 0
                if len(self.work_states[idx].memory) > 0:
                    self.work_states[idx].memory[-1].reward = reward
                    if done or len(self.work_states[idx].memory) == self.local_time_max + 1:
                        self.collect_experience(idx, done)
                action, value, action_prob = self.predict(state)
                self.work_states[idx].memory.append(
                    TransitionExperience(state, action, reward=None, value=value, prob=action_prob))
                self.remotes[idx].send(action)

    def predict(self, state):
        logit, value = self.model(np.array([state], dtype=np.float32))
        policy = tf.nn.softmax(logit).numpy()[0]
        action = np.random.choice(self.action_size, p=policy)
        return action, value.numpy()[0], policy[action]

    def collect_experience(self, idx, done):
        mem = self.work_states[idx].memory
        if not done:
            R = mem[-1].value[0]
            last = mem[-1]
            mem = mem[:-1]
        else:
            R = 0
        mem.reverse()
        for k in mem:
            R = np.clip(k.reward, -1, 1) + self.gamma * R
            # R = k.reward + self.gamma * R
            self.queue.put([k.state, k.action, R, k.prob])
        if not done:
            self.work_states[idx].memory = [last]
        else:
            self.work_states[idx].memory = []

    def record(self, step, **kwargs):
        if step % 100 == 0:
            train_mean_score = np.mean(self.scores) if len(self.scores) > 1 else 0.0
            kwargs["train_mean_score"] = train_mean_score
            log_txt = f"Step:{step}, " + ','.join([f" {k}:{v}" for k, v in kwargs.items()])
            print(log_txt + "," + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.train_summary(step=step, **kwargs)
        if step % 18000 == 0:
            scores = [self.test_env() for _ in range(50)]
            mean_score, max_score = np.mean(scores), np.max(scores)
            logging.info("Mean Score: {}, Max Score: {}".format(np.mean(scores), np.max(scores)))
            self.train_summary(step=step, mean_score=mean_score, max_score=max_score)
        if step % 6000 == 0:
            self.checkpoint_save(step // 6000 % 5)




class Workers(Process):
    def __init__(self, idx: int, master_conn, worker_conn, env_id):
        super().__init__()
        self.idx = idx
        self.name = 'worker-{}'.format(self.idx)
        self.master_conn = master_conn
        self.worker_conn = worker_conn
        self.env_id = env_id

    def run(self):
        env = self.get_env()
        state = env.reset()
        reward, done = 0, False
        while True:
            self.worker_conn.send((self.idx, state, reward, done))
            action = self.worker_conn.recv()
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()

    def get_env(self):
        if self.env_id.startswith("CartPole"):
            return gym.make("CartPole-v0")
        return make_atari(env_id=self.env_id, max_episode_steps=60000)


if __name__ == '__main__':
    configs = {
        'nenvs': mp.cpu_count() * 2,
        'lr': 0.001,
        'discount_gamma': 0.99,
        'batch_size': 128,
        'local_time_max': 5,
        'step_max': 1e9,
        'eval_episodes': 50,
    }

    agent = Master(env_id="SpaceInvaders-v0", **configs)
    agent.learn()

    # agent.play(5, model_path="/home/geb/PycharmProjects/drlgeb/drlgeb/ac/train_logs/train-SpaceInvaders-v0-20210120-214933")
