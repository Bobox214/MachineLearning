import gym
import gym.spaces
import random
import time
import collections
import numpy as np

from tensorboardX import SummaryWriter

class Agent(object):
    params = {
        'EPSILON_DECAY' : 1000
    ,   'EPSILON_MIN'   : 0.2
    }
    def __init__(self,env,params=None):
        self.env = env
        self.params = self.params.copy()
        if params is not None:
            self.params.update(params)
        self.epsilon = 1.0
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.reward = 0
        self.epsilon = max(self.params['EPSILON_MIN'],self.epsilon-1/self.params['EPSILON_DECAY'])
    
    def step(self):
        old_state = self.state
        if random.random()<self.epsilon:
            # random action
            action = 3 if random.random()<0.5 else 2
        else:
            # best action predicted by the NN
            action = 2

        self.state,reward,done,_ = env.step(action)
        self.reward += reward
        if done:
            return self.reward
        else:
            return None

    def render(self):
        self.env.unwrapped.render()

env = gym.make("PongNoFrameskip-v4")
print('Env:',env)
print('Observation',env.unwrapped.observation_space)
print('Action',env.unwrapped.action_space)
print('Action meanging',env.unwrapped.get_action_meanings())

agent = Agent(env)

frame_idx = 0
game_idx  = 0
ts = time.time()
total_reward = 0

total_rewards = collections.deque(maxlen=100)
best_mean = -np.inf

while True:
    frame_idx += 1
    reward = agent.step()
    agent.render()
    if reward is not None:
        # game is done
        game_idx += 1
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards)
        if game_idx>=10 and mean_reward>best_mean:
            best_mean = mean_reward

        print(f'Game:{game_idx}, Frame: {frame_idx}, Reward: {reward}, Mean: {mean_reward:.3f}, Best:{best_mean:.3f}, Epsilon:{agent.epsilon}, Speed {frame_idx/(time.time()-ts):.0f}f/s')
        agent.reset()

print(f'Epsilon: {agent.epsilon}')
env.close()