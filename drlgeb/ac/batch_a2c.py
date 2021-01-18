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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_rate, l: list):
        super(CustomSchedule, self).__init__()
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
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class Master(Agent):
    class WorkerState(object):
        def __init__(self):
            self.memory = []  # list of Experience
            self.score = 0

    def __init__(self, env_id="SpaceInvaders-v0", model='cnn'):
        self.env = make_atari(env_id=env_id)
        self.env_id = env_id
        self.nenvs = mp.cpu_count() * 2
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        if model == 'cnn':
            self.model = ActorCriticModel(self.state_shape, self.action_size)
        else:
            self.model = DenseAC(self.state_shape, self.action_size)
        self.lr = CustomSchedule(0.001, [(15360000, 0.0003), (92160000, 0.0001)])
        self.opt = tf.keras.optimizers.Adam(self.lr, epsilon=1e-3)
        self.local_time_max = 5
        self.gamma = 0.99
        self.batch_size = mp.cpu_count() * 8
        self.step_max = 10000000
        self.episode = 0
        self.scores = deque(maxlen=100)
        super().__init__(name=env_id)

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
        batch = 0
        step = 0
        while step < self.step_max:
            states = []
            actions = []
            discount_returns = []
            action_probs = []
            while True:
                state, action, R, action_prob = self.queue.get()
                step += 1
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
                        loss = tf.add_n([policy_loss, entropy_loss * (0.01 if step < 61440000 else 0.005),
                                         value_loss*0.5]) / self.batch_size

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    grads = [(tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))) for grad in grads]
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    batch += 1

                    self.train_summary(step=step, loss=loss)

                    if batch % 50 == 0:
                        templete = "Batch {}, step {}, pred_reward: {}, loss:{}, policy_loss: {}, entropy_loss:{}, value_loss:{}, importance:{}， train_score:{}"
                        print(templete.format(batch, step, pred_reward, loss, policy_loss, entropy_loss, value_loss,
                                              tf.reduce_mean(importance), np.mean(self.scores)))
                    if batch % 500 == 0:
                        scores = [self.test_env() for _ in range(10)]
                        mean_score, max_score = np.mean(scores), np.max(scores)
                        print("=" * 50)
                        print("Mean Score: {}, Max Score: {}".format(np.mean(scores), np.max(scores)))
                        print("=" * 50)
                        self.train_summary(step=step, mean_score=mean_score, max_score=max_score)
                    if batch % 1000 == 0:
                        self.checkpoint_save()
                    break

    def learn(self):

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.work_states = [self.WorkerState() for _ in range(self.nenvs)]
        self.ps = [Workers(i, remote, work_remote, self.env_id) for i, (remote, work_remote) in
                   enumerate(zip(self.remotes, self.work_remotes))]

        self.queue = queue.Queue(maxsize=self.batch_size * 2 * 8)

        for work in self.ps:
            print(f"{work.name} Start!")
            work.start()
        # self.ts = [threading.Thread(target=self.recv_send, args=(i,)) for i in range(self.nenvs)]
        # for t in self.ts:
        #     t.start()

        t = threading.Thread(target=self.recv_send)
        t.start()

        update = threading.Thread(target=self.update())
        update.start()



        [work.join() for work in self.ps]
        t.join()
        update.join()
        # [t.join() for t in self.ts]

    def recv_send(self):
        while True:
            idxs = np.random.choice(range(self.nenvs), self.batch_size)
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
    agent = Master()

    agent.learn()
