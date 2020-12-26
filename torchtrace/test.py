'''
@Author: your name
@Date: 2020-06-15 01:25:07
@LastEditTime: 2020-06-15 15:32:26
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \ML-project-2020\Lunarlander\Tracer\test.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tracer import tracer
from .optim import RMSprop
from .nn import Sequential, Linear, ReLU, MSE, Gather_last, Sum, Model, Conv2d
from . import load, save
from torchsummary import summary

class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64,4)
        self.f = nn.ReLU()

    def forward(self,x):
        x =  self.fc1(x)
        x =  self.f(x)
        x =  self.fc2(x)
        x =  self.f(x)
        x = self.fc3(x)
        x = self.f(x)
        x = self.fc4(x)
        return x
class Test_conv(nn.Module):
    def __init__(self):
        super(Test_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(64, 256, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(256, 128, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 128, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
    
class myConv(Model):
    def __init__(self):
        self.conv = Sequential(
            Conv2d(3, 64, kernel_size=8, stride=4), ReLU(),
            Conv2d(64, 256, kernel_size=4, stride=2), ReLU(),
            Conv2d(256, 128, kernel_size=3, stride=1), ReLU(),
        )  
        self.fc = Linear(7*7*128, 1)
        super(myConv, self).__init__()
        
    def construct(self):
        ops = [self.conv, self.fc]
        return ops
        
    def forward(self, x):
        x = self.conv(x)
        x = x.View(x.shape[0], -1)
        return self.fc(x)
    
def test_Linear():
    # -------------------------------------
    seed = 0
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # -------------------------------------
    test = Test_model()
    model = Sequential(
        Linear(8,256),
        ReLU(),
        Linear(256,128),
        ReLU(),
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    # print(test)
    print(model)
    for para in test.parameters():
        para.data.fill_(0.1)
    for para in model.parameters():
        para.data.fill_(0.1)
    # print("INIT model parameters as 1 ")
    
    x = torch.rand(32,8).float()
    another = torch.rand(32,4)
    mse = MSE()
    
    test_out = F.mse_loss(test(x), another)
    model_out = mse(model(x), another)
    # print(test_out)
    # print(model_out)
    test_out.backward()
    model_out.backward(show=True)
    print(torch.typename(test.fc4.weight.grad), torch.typename(model.ops[-1].weight.grad))
    print("Grad errs:", torch.mean(torch.abs(test.fc4.weight.grad - model.ops[-1].weight.grad)))
    print("Grad errs:", torch.mean(torch.abs(test.fc3.weight.grad - model.ops[-3].weight.grad)))
    print("Grad errs:", torch.mean(torch.abs(test.fc2.weight.grad - model.ops[-5].weight.grad)))
    print("Grad errs:", torch.mean(torch.abs(test.fc1.weight.grad - model.ops[-7].weight.grad)))
    
def test_load():
    model1= Sequential(
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    model2 = Sequential(
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    for para in model1.parameters():
        para.data.fill_(0.1)
    for para in model2.parameters():
        pass
    model2.load_seq_list(model1.seq_list())
    for para in model2.parameters():
        print(para)
    
def test_gather():
    model = Sequential(
        Linear(8,256),
        ReLU(),
        Linear(256,128),
        ReLU(),
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    print(model)
    x = torch.rand(16,8).float()
    target = torch.rand(16,1)
    mse = MSE()
    
    out = model(x)
    _, action = out.max(1)
    # gather = Gather_last(action)
    # out = gather(out)
    out = out.Gather(action)
    
    loss = mse(out, target)
    loss.backward(show=True)
    print(loss)
    
def test_optim():
    model = Sequential(
        Linear(8,256),
        ReLU(),
        Linear(256,128),
        ReLU(),
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    #model = Test_model()
    optimizer = RMSprop(model.parameters(), lr=5e-4)
    #optimizer = torch.optim.RMSprop(model_torch.parameters(), lr=5e-4)
    x = torch.rand(128,8).float()
    another = torch.rand(128,4)
    mse = MSE()
    #mse = F.mse_loss
    for i in range(10):
        model_out = mse(model(x), another)
        print(model_out)
        optimizer.zero_grad()
        model_out.backward()
        optimizer.step()
    
def test_load_save():
    import os
    model = Sequential(
        Linear(8,256),
        ReLU(),
        Linear(256,128),
        ReLU(),
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
    save('see.pth', model.seq_list())
    model.load_seq_list(load('see.pth'))
    os.remove('see.pth')
    
def test_conv():
    from .nn import Conv2d
    x = torch.rand(3, 3, 84, 84).float()/100
    model_torch = Test_conv()
    model_trace = myConv()
    summary(model_torch, input_size=(3,84,84))
    # print(model_trace)
    for name, para in model_torch.named_parameters():
        para.data.fill_(0.01)
    for para in model_trace.parameters():
        para.data.fill_(0.01)
    out1 = model_torch(x)
    out2 = model_trace(x)
    out1 = out1.sum()
    out2 = out2.Sum()
    print("torch out:", out1)
    print("trace out:", out2)
    out1.backward()
    out2.backward(show=True)
    children = [module for module in model_torch.conv.children()]
    print("Grad error", torch.mean(torch.abs(model_torch.fc.weight.grad - model_trace.fc.weight.grad)))
    # print(children[-2].weight.grad)
    # print('my',  model_trace.conv.ops[-2].weight.grad)
    print("Grad error", torch.mean(torch.abs(children[-2].weight.grad - model_trace.conv.ops[-2].weight.grad)))
    print("Grad error", torch.mean(torch.abs(children[-4].weight.grad - model_trace.conv.ops[-4].weight.grad)))
    print("Grad error", torch.mean(torch.abs(children[-6].weight.grad - model_trace.conv.ops[-6].weight.grad)))
    
if __name__ == "__main__":
    # test_Linear()
    # test_load()
    # test_gather()
    # test_optim()
    # test_load_save()
    from time import time
    start = time()
    test_conv()
    print("TIME", time() - start)
    