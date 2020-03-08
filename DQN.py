# Cartpole 게임
# state: cart position, cart velocity, pole angle, pole velocity
# action: moving left, moving right

# step이 지날 때마다 +1 reward
# pole의 각도가 +- 12도를 넘거나 cart position이 +- 2.4를 넘으면
# pole은 쓰러지고 게임은 종료된다.
# gym 은 200 에피소드 이상을 허락하지 않았다.

import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states, ))
        self.hidden_layers = []

        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=i, activation='tanh', kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(
            units=num_actions, activation='linear', kernel_initializer='RandomNormal')

    # forward pass
    @tf.function
    def call(self, inputs):
        # inputs shape: (batch_size, num_states)
        # output shape: (batch_size, num_actions)
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma,
                 max_experiences, min_experiences, batch_size, learning_rate):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model = MyModel(num_states, hidden_units, num_actions)

        # 경험 리플레이 -- Experience Replay Dataset: buffer
        self.experience = {'s': [], 'a': [], 'r': [], 'next_s': [], 'done': []}

        # Agent는 min_experiences 이상이어야 학습을 진행하며
        # max_experiences가 buffer의 size보다 크면 가장 old한 값을 삭제하여 공간을 확보한다.
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        # inputs는 single state 일수도 batch of states 일수도 있음
        # 아래에 atleast_2d는 만약 single state가 들어왔을 때 2차원으로 변형해주는 함수임
        return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        # 1회 학습에 대한 메서드

        # 100개의 experience가 모이지 않으면, 그냥 아래 학습을 진행하지 않고 멈춘다. 아래 0은 아무 의미 없다.
        if len(self.experience['s']) < self.min_experiences:
            return 0

        # batch_size 만큼 학습을 할 데이터를 샘플링하는 작업임
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        next_states = np.asarray([self.experience['next_s'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        next_value = np.max(TargetNet.predict(next_states), axis=1)
        true_value = np.where(dones, rewards, rewards + self.gamma*next_value)

        # true_value = y = 타겟 값
        # estimated_value = selected_action_value = 추정 값 = y_hat
        with tf.GradientTape() as tape:
            # self.predict(states) shape: (batch_size, num_actions=2)
            # tf.one_hot(actions, num_actions) shape: (batch_size, num_actions=2)
            # estimated_value shape: (batch_size, )

            # actions = 현재 시점에서의 action 값(들)
            # 아래 두 값을 곱하면 actions에 선택되지 못한 action에 해당하는 predict(states)의 결과값은 0처리가 됨
            # reduce_sum을 하면 이 값들을 합쳐서 실수로 만들어주고, 이를 연결하여 batch_size 길이를 만듦
            estimated_value = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(true_value - estimated_value))

        variables = self.model.trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)

        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

"""
model = MyModel(num_states=4, hidden_units= [200, 200], num_actions=2)
env = gym.make('CartPole-v0').env
observations = env.reset()
observations = observations.reshape(1, 4).astype('float32')
output = model(inputs=observations)
"""

# Play a Game
def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    # 한 에피소드가 끝날 때까지 게임을 하는 것
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()

    while not done:
        action = TrainNet.get_action(states=observations, epsilon=epsilon)
        previous_observations = observations

        observations, reward, done, _ = env.step(action)
        rewards += reward

        if done:
            reward = -200
            env.reset()

        exp = {'s': previous_observations, 'a': action, 'r': reward, 'next_s': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)

        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards


# Make a video for testing
def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observations = env.reset()
    while not done:
        action = TrainNet.get_action(states=observations, epsilon=0)
        observations, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward

    print("Testing steps: {} / rewards: {}".format(steps, rewards))



def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    # 총 N개의 에피소드를 진행한다.
    N = 10000
    total_rewards = np.empty(N)    # shape = (N, )
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1

    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)

        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)

    print("avg reward for last 100 episodes:", avg_rewards)
    make_video(env, TrainNet)
    env.close()


if __name__ == '__main__':
    main()

