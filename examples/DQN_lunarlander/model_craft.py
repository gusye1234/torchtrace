
import torch
from torchtrace.nn import Model
import torchtrace.nn as nn
class myDQN_network(Model):
    def __init__(self, 
                 state_size,
                 action_size):
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.out = nn.Linear(64, action_size)
        self.f   = nn.ReLU
        super(myDQN_network, self).__init__()
    
    def construct(self):
        ops = [self.fc1,
               self.f(),
               self.fc2,
               self.f(),
               self.fc3,
               self.f(),
               self.out]
        return ops
    
    def forward(self, x):
        for block in self.ops:
            x = block(x)
        return x
    
    
if __name__ == "__main__":
    model = myDQN_network(8,4)
    print(model)
    x = torch.randn(16,8)
    model(x)