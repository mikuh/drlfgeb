import numpy as np
import tensorflow as tf
from multiprocessing import Process, Pipe
import gym
import threading
import queue
from collections import deque
from drlgeb.common import make_atari
from drlgeb.ac.model import ActorCriticModel


# class ActorCriticModel(tf.keras.Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCriticModel, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.dense1 = tf.keras.layers.Dense(128, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#
#         self.policy_logits = tf.keras.layers.Dense(action_size)
#         self.values = tf.keras.layers.Dense(1)
#
#     def call(self, inputs):
#         x = self.dense1(inputs)
#         v1 = self.dense2(x)
#         logits = self.policy_logits(x)
#         values = self.values(v1)
#         return logits, values


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class Master(object):
    class WorkerState(object):
        def __init__(self):
            self.memory = []  # list of Experience
            self.score = 0

    def __init__(self):
        env = make_atari(env_id="SpaceInvaders-v0")
        self.nenvs = 8
        self.state_shape = env.observation_space.shape
        self.action_size = env.action_space.n
        del env
        self.model = ActorCriticModel(self.state_shape, self.action_size)
        self.opt = tf.keras.optimizers.Adam(lr=0.001)
        self.local_time_max = 5
        self.gamma = 0.99
        self.batch_size = 128
        self.step_max = 10000000
        self.episode = 0
        self.socres = deque(maxlen=100)

    def learn(self):

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.work_states = [self.WorkerState() for _ in range(self.nenvs)]
        self.ps = [Workers(i, remote, work_remote) for i, (remote, work_remote) in
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
                        # advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)))

                        loss = tf.add_n([policy_loss, entropy_loss * 0.01, value_loss * 0.5]) / discount_returns.shape[0]

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    break
                epoch = int(step / 6000 + 1)
                if len(self.socres) > 10 and step % 200 == 0:
                    print(
                        f"epoch {epoch}, step {step}, pred_reward: {pred_reward}, mean_reward: {np.mean(self.socres)}")

        # [work.join() for work in self.ps]
        # [t.join() for t in self.ts]

    def recv_send(self):
        while True:
            for idx in range(self.nenvs):
                work_idx, state, reward, done = self.remotes[idx].recv()
                if len(self.work_states[idx].memory) > 0:
                    self.work_states[idx].memory[-1].reward = reward
                    self.work_states[idx].score += reward
                    if done:
                        self.socres.append(self.work_states[idx].score)
                        self.work_states[idx].score = 0
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
            R = mem[-1].value
            last = mem[-1]
            mem = mem[:-1]
        else:
            R = 0
        mem.reverse()
        for k in mem:
            # R = np.clip(k.reward, -1, 1) + self.gamma * R
            R = k.reward + self.gamma * R
            self.queue.put([k.state, k.action, R, k.prob])
        if not done:
            self.work_states[idx].memory = [last]
        else:
            self.work_states[idx].memory = []


class Workers(Process):
    def __init__(self, idx: int, master_conn, worker_conn):
        super().__init__()
        self.idx = idx
        self.name = 'worker-{}'.format(self.idx)
        self.master_conn = master_conn
        self.worker_conn = worker_conn

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
        # return gym.make("CartPole-v0")
        return make_atari(env_id="SpaceInvaders-v0", max_episode_steps=60000)

if __name__ == '__main__':
    agent = Master()

    agent.learn()
