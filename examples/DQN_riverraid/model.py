import sys
sys.path.append('..')
sys.path.append('../..')

import torchtrace
import torchtrace.nn as nn
import random
from collections import deque
class DQN_craft(nn.Model):
    def __init__(self):
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()).set_name("Conv")
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, 18)).set_name("Full-connect")
        super(DQN_craft, self).__init__()
        self.set_name('DQN')

    def construct(self):
        return [self.conv, self.fc]
    
    def forward(self, obs):
        obs = self.conv(obs)
        obs = obs.View(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])
        actions = self.fc(obs)
        return actions




# --------------------------------------------------------------------------------------------------------------
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
    
if __name__ == "__main__":
    model = DQN_craft()
    print(model)