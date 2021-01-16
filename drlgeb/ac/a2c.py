import tensorflow as tf
import sys
sys.path.append("../../")
import numpy as np
import gym
import random
import time

from drlgeb.ac import ActorCriticModel
from drlgeb.common import make_atari



class Agent(object):

    def __init__(self, model, env):

        self.env = env
        self.action_size = self.env.action_space.n
        self.a2c = model
        self.n_step = 1
        self.repeat_sampling = 1
        self.gamma = 0.99
        self.rollout = 128
        self.batch_size = 128
        self.lr = 0.001
        self.epsilon = 0.5
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        self.episode = 0
        self.score = 0

    def get_action(self, state, is_train=True):
        # state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.a2c(np.array([state]))
        policy = tf.nn.softmax(logits)
        policy = np.array(policy)[0]
        # self.epsilon = 1 / (self.episode * 0.1 + 5)
        # if random.random() < self.epsilon and is_train:
        #     action = np.random.choice(self.action_size)
        # else:
            # where_are_nan = np.isnan(policy)
            # policy[where_are_nan] = 1e-8
            # policy = softmax(policy/temperature)
        action = np.random.choice(self.action_size, p=policy)
        return action

    def collect_replay_buffer(self, state):
        state = np.array(state)
        state_list, next_state_list, reward_list, done_list, action_list = [], [], [], [], []

        for _ in range(self.rollout):
            action = self.get_action(state, False)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.array(next_state)

            self.score += reward

            reward = np.clip(reward, -1, 1)

            state_list.append(state)
            next_state_list.append(next_state)
            reward_list.append(np.float32(reward))
            done_list.append(np.float32(done))
            action_list.append(np.int32(action))

            state = next_state

            if done:
                self.episode += 1
                print("Episode %s, Score: %s,  at: %s" % (
                    self.episode, self.score, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                state = self.env.reset()
                state = np.array(state)
                if self.episode % 100 == 0:
                    self.a2c.save("a2c_model/")
                self.score = 0

        if self.n_step > 1:
            states, next_states, rewards, dones, actions = [], [], [], [], []
            for i in range(len(state_list) - self.n_step + 1):
                state = state_list[i]
                next_state = next_state_list[i + self.n_step - 1]
                reward = 0
                for index, x in enumerate(reward_list[i: i + self.n_step]):
                    reward += self.gamma ** index * x
                done = done_list[i]
                action = action_list[i]

                states.append(state)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                actions.append(action)
            return states, next_states, rewards, dones, actions, state

        return state_list, next_state_list, reward_list, done_list, action_list, state

    def learn(self, max_episode):
        init_state = self.env.reset()
        while self.episode < max_episode:
            _state, _next_state, _reward, _done, _action, init_state = self.collect_replay_buffer(init_state)
            for _ in range(self.repeat_sampling):
                sample_range = np.arange(self.rollout - self.n_step + 1)
                np.random.shuffle(sample_range)
                sample_idx = sample_range[:self.batch_size]

                state = [_state[i] for i in sample_idx]
                next_state = [_next_state[i] for i in sample_idx]
                reward = [_reward[i] for i in sample_idx]
                done = [_done[i] for i in sample_idx]
                action = [_action[i] for i in sample_idx]

                a2c_variable = self.a2c.trainable_variables
                with tf.GradientTape() as tape:
                    tape.watch(a2c_variable)

                    _, current_value = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
                    _, next_value = self.a2c(tf.convert_to_tensor(next_state, dtype=tf.float32))
                    current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)

                    target = tf.stop_gradient(
                        self.gamma * (1 - np.array(done)) * np.array(next_value) + np.array(reward))
                    value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

                    logits, _ = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
                    policy = tf.nn.softmax(logits)
                    entropy = tf.reduce_mean(- policy * tf.math.log(policy + 1e-8)) * 0.1
                    # action = tf.convert_to_tensor(action, dtype=tf.int32)
                    onehot_action = tf.one_hot(action, self.action_size)
                    action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
                    adv = tf.stop_gradient(target - current_value)
                    pi_loss = -tf.reduce_mean(tf.math.log(action_policy + 1e-8) * adv) - entropy

                    total_loss = pi_loss + value_loss

                grads = tape.gradient(total_loss, a2c_variable)
                self.opt.apply_gradients(zip(grads, a2c_variable))

            # print("Complete parameters update at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        self.a2c.save("my_model/")

    def play(self):
        obs = self.env.reset()
        obs = np.array(obs)
        score = 0
        while True:
            action = self.get_action(obs, is_train=False)
            obs, rewards, dones, info = self.env.step(action)
            obs = np.array(obs)
            score += rewards
            self.env.render()
            if dones > 0:
                print("得分:", score)
                obs = self.env.reset()
                obs = np.array(obs)
                score = 0


if __name__ == '__main__':
    
    env = make_atari("SpaceInvaders-v0", max_episode_steps=60000)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # a2c = ActorCriticModel((1,)+state_shape, action_size)
    # a2c = tf.keras.models.load_model("best_model/")
    # agent = Agent(model=a2c, env=env)

    # agent.learn(1000000)

    # play
    a2c = tf.keras.models.load_model("a2c_model/")
    agent = Agent(model=a2c, env=env)
    agent.play()
