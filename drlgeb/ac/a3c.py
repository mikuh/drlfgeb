import os
import sys

sys.path.append("../../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import datetime
from drlgeb.ac import ActorCriticModel
from drlgeb.common import make_atari
from queue import Queue
import numpy as np
import tensorflow as tf
import multiprocessing
import threading


class Memory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class MasterAgent(object):

    def __init__(self, env_id="SpaceInvaders-v0", **configs):
        self.env_id = env_id
        self.configs = configs
        env = make_atari(env_id, max_episode_steps=60000)
        self.state_shape = env.observation_space.shape
        self.action_size = env.action_space.n

        self.opt = tf.keras.optimizers.Adam(lr=configs["lr"])

        self.global_model = ActorCriticModel(
            self.state_shape, self.action_size)  # global network
        self.global_model(np.random.random((1,) + self.state_shape))

    def learn(self):
        res_queue = Queue()

        workers = [Worker(self.state_shape,
                          self.action_size,
                          self.global_model,
                          self.opt,
                          res_queue,
                          i,
                          env_id=self.env_id,
                          configs=self.configs) for i in
                   range(multiprocessing.cpu_count())]  # multiprocessing.cpu_count()

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        [w.join() for w in workers]


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_shape,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 env_id='SpaceInvaders-v0',
                 save_dir='/tmp',
                 configs=None):
        super(Worker, self).__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_shape, self.action_size)
        self.worker_idx = idx
        self.env_id = env_id
        self.env = make_atari(self.env_id, max_episode_steps=60000)
        self.save_dir = save_dir
        self.ep_loss = 0.0

        self.max_eps = configs['max_eps']
        self.gamma = configs['gamma']
        self.update_freq = configs['update_freq']

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_score = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, values = self.local_model(np.array([current_state]))
                policy = tf.nn.softmax(logits, name='policy')

                action = np.random.choice(
                    self.action_size, p=np.array(policy)[0])
                new_state, reward, done, _ = self.env.step(action)
                ep_score += reward
                mem.store(current_state, action, reward)

                if time_count == self.update_freq or done:
                    with tf.GradientTape() as tape:
                        loss = self.get_loss(done, new_state, mem)
                        self.ep_loss += loss
                        # Calculate local gradients
                        grads = tape.gradient(
                            loss, self.local_model.trainable_weights)
                        # Push local gradients to global model
                        self.opt.apply_gradients(
                            zip(grads, self.global_model.trainable_weights))
                        # Update local model with new weights
                        self.local_model.set_weights(
                            self.global_model.get_weights())
                    mem.clear()
                    time_count = 0
                if done:
                    Worker.global_episode += 1
                    # current_time = datetime.datetime.now().strftime("%Y%m%d% H%:M%:S")
                    print(
                        f"Episode: {Worker.global_episode}, Score: {ep_score}, work: {self.worker_idx}")
                    if Worker.global_episode % 200 == 0:
                        self.global_model.save("my_model/")
                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1

        self.result_queue.put(None)

    def get_loss(self,
                 done,
                 new_state,
                 memory):
        if done:
            reward_sum = 0.0
        else:
            _, values = self.local_model(np.array([new_state]))
            reward_sum = values.numpy()[0][0]
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = np.clip(reward, -1, 1) + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        states = np.array(memory.states)
        logits, values = self.local_model(states)
        advantage = np.array(discounted_rewards)[:, None] - values

        value_loss = tf.nn.l2_loss(advantage)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss = policy_loss * tf.stop_gradient(advantage) - 0.01 * entropy
        loss = tf.reduce_mean((1 * value_loss + policy_loss))
        return loss


if __name__ == '__main__':
    lr = 0.001
    max_eps = 100000
    gamma = 0.99
    update_freq = 100

    agent = MasterAgent(lr=lr, max_eps=max_eps,
                        gamma=gamma, update_freq=update_freq)

    agent.learn()
