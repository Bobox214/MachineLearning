import gym
import gym.spaces
import random
import time
import collections
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from tensorflow.keras        import backend as K
from tensorflow.keras.models import Model,clone_model

from tensorboardX import SummaryWriter

def NN(X,params):
    #X = layers.Dense(512,activation='relu',kernel_initializer='zeros')(X)
    #X = layers.Dense(params['n_actions'],kernel_initializer='zeros')(X)
    X = layers.Dense(512,activation='relu')(X)
    X = layers.Dense(params['n_actions'])(X)
    return X

Experience = collections.namedtuple('Experience',field_names=('state','action','reward','is_done','new_state'))

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,high=1,shape=(env.observation_space.n,),dtype=np.uint8)
        self.n = env.observation_space.n
    
    def observation(self,obs):
        return [int(x==obs) for x in range(self.n)]
def agent_mse(y_pred,y_true):
    return losses.mean_squared_error(y_pred,y_pred[:,0])

class Agent(object):
    params = {
        'EPSILON_DECAY'          : 10**4
    ,   'EPSILON_MIN'            : 0.02
    ,   'EXPERIENCE_BUFFER_SIZE' : 100000
    ,   'SYNC_TARGET_MODEL'      : 1024
    ,   'BATCH_SIZE'             : 32
    ,   'GAMMA'                  : 0.99
    }
    def __init__(self,envName,params=None):
        self.params = self.params.copy()
        if params is not None:
            self.params.update(params)
        env = gym.make(envName)
        env = OneHotWrapper(env)
        self.env = env
        self.n_action = env.action_space.n
        self.build_models()
        self.epsilon = 1.0
        self.experience_buffer = collections.deque(maxlen=self.params['EXPERIENCE_BUFFER_SIZE'])
        self.reset()
        self._step_count = 0

    def build_models(self):
        I = layers.Input(shape=self.env.observation_space.shape)
        X = NN(I,{'n_actions':self.n_action})
        model = Model(inputs=I,outputs=X)
        model.summary()

        model.compile(optimizer='adam',loss='mse')
        self.model = model
        self.tgt_model = clone_model(self.model)

    def reset(self):
        self.state = self.env.reset()
        self.reward = 0
        self.epsilon = max(self.params['EPSILON_MIN'],self.epsilon-1/self.params['EPSILON_DECAY'])
    
    def step(self):
        self._step_count += 1
        old_state = self.state
        if random.random()<self.epsilon:
            # random action
            action = random.randint(0,self.n_action-1)
        else:
            # best action predicted by the NN
            action = np.argmax(self.model.predict(np.expand_dims(self.state,0))[0])

        self.state,reward,is_done,_ = self.env.step(action)
        self.experience_buffer.append(Experience(old_state,action,reward,is_done,self.state))
        self.reward += reward
        if self._step_count % self.params['SYNC_TARGET_MODEL'] == 0:
            self.train_model()
            self.model.save_weights('h5/FrozenLake-run.h5')
            self.tgt_model.load_weights('h5/FrozenLake-run.h5')
        #if self._step_count > self.params['EXPERIENCE_BUFFER_SIZE']:
        if is_done:
            return self.reward
        else:
            return None

    def train_model(self):
        sample_size = self.params['SYNC_TARGET_MODEL']
        indices = np.random.choice(len(self.experience_buffer),sample_size,replace=False)
        states,actions,rewards,dones,next_states = zip( *(self.experience_buffer[idx] for idx in indices))
        if sum(rewards)==0: return
        #states,actions,rewards,dones,next_states = zip(*self.experience_buffer)
        next_state_rewards = np.max(self.tgt_model.predict(np.array(next_states)),axis=1)
        next_state_rewards[np.array(dones)] = 0
        y = self.tgt_model.predict(np.array(states))
        exp_rewards = np.array(rewards) + next_state_rewards*self.params['GAMMA']
        for i,action in enumerate(actions):
            y[i][action] = exp_rewards[i]
        #print('w',w)
        #print('a',actions)
        #print('e',exp_rewards)
        #print('y',y)
        self.model.fit(x=np.array(states),y=y,batch_size=self.params['BATCH_SIZE'],epochs=10,verbose=False)
        #n = self.env.unwrapped.observation_space.n
        #print('Exp rewards',exp_rewards)
        #for i in range(n):
        #    obs = [ int(x==i) for x in range(n) ]
        #    print(f'\t{i}:',self.model.predict(np.expand_dims(obs,0))[0])
        #time.sleep(1)
        
    def render(self):
        self.env.unwrapped.render()
    
    def close(self):
        self.env.close()


agent = Agent("FrozenLake-v0")
print('Env:',agent.env.unwrapped)
print('Observation',agent.env.unwrapped.observation_space)
print('Action',agent.env.unwrapped.action_space)

frame_idx = 0
game_idx  = 0
ts = time.time()
total_reward = 0

total_rewards = collections.deque(maxlen=100)
best_mean = -np.inf

while True:
    frame_idx += 1
    reward = agent.step()
    #agent.render()
    if reward is not None:
        # game is done
        game_idx += 1
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards)
        if game_idx>=10 and mean_reward>best_mean:
            best_mean = mean_reward
            if mean_reward > 0.8:
                print(f"Solved in {frame_idx} frames, Speed {frame_idx/(time.time()-ts):.0f}f/s")
                break

        print(f'Game:{game_idx}, Frame: {frame_idx}, Reward: {reward}, Mean: {mean_reward:.3f}, Best:{best_mean:.3f}, Epsilon:{agent.epsilon:.2f}, Speed {frame_idx/(time.time()-ts):.0f}f/s')
        agent.reset()

print(f'Epsilon: {agent.epsilon}')
agent.close()