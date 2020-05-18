
# 목표
# 승객을 적절한 위치에 내려준다.
# 승객을 내려줄 때 최소한의 시간이 걸리게 한다.
# 교통 수칙을 지킨다.

# 1. Rewards
# 승객을 제대로 내려주면 큰 보상을 받고, 그렇지 않으면 벌을 받는다.
# 매 time step 마다 Agent는 아직 도착하지 못한 것에 대해 작은 negative reward를 얻는다.

# 2. State Space
# 5X5 grid.
# (R, G, Y, B): 4개의 목적지 가능

# 가능한 State의 수
# 5 X 5 X 5 X 4 = 500 total possible states
# Grid 가로 X Grid 세로 X (가능한 승객 위치 수) X (가능한 목적지 수)
# 승객은 R, G, Y, B 외에도 택시 내부에 있을 수도 있기 때문에 총 5개의 가능성을 가진다.

# 3. Action Space
# 동서남북 + pickup + dropoff = 6가지
# 벽에 부딪히면 -1 페널티를 줌

import gym

env = gym.make('Taxi-v3').env

# env.reset: Resets the environment and returns a random initial state
# env.step(action): step the environment by one time step
# - return: observation, reward, done, info
# -- done: 제대로 승객을 태웠는지/내리게 했는지 나타냄, 한 episode 의 끝을 의미함
# env.render: 이미지로 현재 상태를 보여줌

# 제대로 dropoff 하면 20포인트 획득, 매 time step 마다 1포인트씩 잃음
# 잘못된 곳에서 pickup/dropoff 하면 -10포인트
# 파란 글씨가 현재 승객의 위치, 보란 글씨가 목적지
# 0(남), 1(북), 2(동), 3(서), 4(pickup), 5(dropoff)

env.reset()
env.render()

print(env.action_space)
print(env.observation_space)

state = env.encode(3, 1, 2, 0)
print(state)
env.s = state
env.render()

# Reward Table: P
# Matrix (states X actions)
# 위에서 본 328번 state에 대한 Reward Table 확인
# {action: [probability, next state, reward, done]
# 이 environment 에서는 언제나 probability 는 1.0 이다.
print(env.P[328])


# Q-learning
# Q-table은 Q-values를 저장하는데, (state, action) 조합에 매핑된다.
# 특정한 (state, action) 조합에 대한 Q-value를 그 state에서의 action에 대한 quality라고 한다.
# state 수 X action 수 = 500 X 6
# 처음에는 모든 원소가 0으로 초기화되며, 점점 업데이트 된다.


# Implement
import random
import numpy as np

# Q 테이블 초기화
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1    # 이보다 작으면 exploration

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1


print(q_table[328])    # 정답인 북쪽의 값이 가장 크다.







