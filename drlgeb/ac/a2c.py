import tensorflow as tf
from drlgeb.common import SubprocVecEnv
from drlgeb.common import make_atari
from drlgeb.ac.model import ActorCriticModel
import multiprocessing as mp
import gym
import numpy as np

import matplotlib.pyplot as plt

# num_envs = mp.cpu_count() * 2
num_envs = 26

def make_env(env_id):
    def _thunk():
        if env_id.startswith("CartPole"):
            env = gym.make(env_id)
        else:
            env = make_atari(env_id)
        return env

    return _thunk


# class ActorCriticModel(tf.keras.Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCriticModel, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.dense1 = tf.keras.layers.Dense(100, activation='relu')
#         self.policy_logits = tf.keras.layers.Dense(action_size)
#         self.dense2 = tf.keras.layers.Dense(100, activation='relu')
#         self.values = tf.keras.layers.Dense(1)
#
#     def call(self, inputs):
#         # Forward pass
#         x = self.dense1(inputs)
#         logits = self.policy_logits(x)
#         v1 = self.dense2(inputs)
#         values = self.values(v1)
#         return logits, values


def plot(frame_idx, rewards):
    plt.plot(rewards, 'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)


class Memory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def store(self, state, action, reward, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []


class AgentMaster(object):

    def __init__(self, env_id="CartPole-v0", **configs):
        envs = [make_env(env_id) for _ in range(num_envs)]
        self.env = make_atari(env_id)
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.envs = SubprocVecEnv(envs)

        self.model = ActorCriticModel(self.state_shape, self.action_size)
        self.lr = configs['lr']
        self.gamma = 0.99
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        self.max_steps = 10000000
        self.n_step = 5
        self.best_score = 0

    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        logits, _ = self.model(state)
        policy = tf.nn.softmax(logits).numpy()[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def count_returns(self, new_values, memory):
        R = new_values
        discounted_returns = []
        for step in reversed(range(len(memory.rewards))):
            R = memory.rewards[step] + self.gamma * R * memory.masks[step]
            discounted_returns.insert(0, R)
        return discounted_returns

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

    def learn(self):
        states = self.envs.reset()
        step = 0
        test_scores = []
        memory = Memory()
        while step < self.max_steps:
            memory.clear()
            for _ in range(self.n_step):
                logits, values = self.model(np.array(states, dtype=np.float32))
                policys = tf.nn.softmax(logits)
                actions = tf.random.categorical(policys, 1)
                actions = tf.squeeze(actions).numpy()
                new_states, rewards, dones, _ = self.envs.step(actions)
                masks = np.array(1 - dones, dtype=np.int32)[:, None]
                memory.store(states, actions, np.clip(np.array(rewards), -1, 1)[:, None],  masks)

                states = new_states
                step += 1

                if step % 500 == 0:
                    # test_scores.append(np.mean([self.test_env() for _ in range(10)]))
                    score = np.mean([self.test_env() for _ in range(10)])
                    if score > 500 and score > self.best_score:
                        self.model.save("SpaceInvaders_model/")
                    if score > self.best_score:
                        self.best_score = score
                    print(f"step:{step}, mean score:{score}, best score: {self.best_score}")

            _, new_values = self.model(np.array(new_states, dtype=np.float32))
            discounted_returns = self.count_returns(new_values, memory)
            discounted_returns = tf.stop_gradient(np.concatenate(discounted_returns))



            with tf.GradientTape() as tape:



                # values = tf.concat(memory.values, 0)
                logits, values = self.model(tf.concat(memory.states, 0))

                actions = tf.concat(memory.actions, 0)
                # logits = tf.concat(memory.logits, 0)
                # probs = tf.concat(memory.probs, 0)
                probs = tf.nn.softmax(logits)
                # print(discounted_returns.shape, values.shape, actions.shape, logits.shape, probs.shape)
                # (40, 1) (40, 1) (40, 1) (40, 2) (40, 2)

                policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
                entropy = tf.reduce_sum(probs * tf.math.log(probs + 1e-20), axis=1)
                advantage = discounted_returns - values
                value_loss = tf.nn.l2_loss(advantage)
                policy_loss = policy_loss * tf.stop_gradient(advantage) - 0.01 * entropy
                loss = tf.reduce_mean((1 * value_loss + policy_loss))

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == '__main__':
    agent = AgentMaster(env_id="SpaceInvaders-v0", lr=0.001)

    agent.learn()
