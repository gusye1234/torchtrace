# torchtrace
Suit for the new learners to have a basic understanding what is happending behind the library with auto-grad supported like `tensorflow`, `torch`.

The basic data type in `torchtrace` is `tracer`, inherited from `torch.Tensor`.Thanks to that, all the arithmetic operation, type conversion are not in our concern.

Note that, all the auto-grad function is **implemented in `torchtrace` itself**, we only use a few traits of `torch`, see the `torch` apis we used in `Whitelist.txt`(exclude`torchtrace.test`, which uses auto-grad traits in `torch` for unit-test ) .

### What is good in `torchtrace`

* Thanks to the `torch`-like api's fashion, we can easily migrate a `torch` code into a `torchtrace` code, if all the operations are supported in `torchtrace`.
* For the operation that are not supported in `torchtrace` right now, we can easily implement one. Please see the implementation in `torchtrace.nn` for details.
* `torchtrace` is easy and easy to read. Right now, `torchtrace` only consider the sequential neural network, which makes the dynamic computation graph pretty straightforward. See more in `torchtrace.base`

### Code snippets

A.

```python
import torchtrace
from torchtrace import nn

a = torchtrace.tracer([1,2,3,4])
a.requires_grad_()
b = a.Sum()
b.backward()
print(a.grad)

# output
# tensor([1., 1., 1., 1.])
```


B.

```python
import torch
import torchtrace
import torchtrace.nn as nn
from torchtrace.nn import Linear, ReLU
import torchtrace.optim as optim

model = nn.Sequential(
        Linear(8,256),
        ReLU(),
        Linear(256,128),
        ReLU(),
        Linear(128,64),
        ReLU(),
        Linear(64, 4),
    )
print(model)
optimizer = optim.RMSprop(model.parameters(), lr=5e-4)
mse = nn.MSE()

x = torch.rand(16,8).float()
target = torch.rand(16,1)

for i in range(10):
        out = model(x)
        _, action = out.max(1)
        out = out.Gather(action)
        loss = mse(loss, target)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

C.

```python
import torch
from torchtrace.nn import Conv2d, ReLU, Linear, Model, Sequential

class myConv(Model):
    def __init__(self):
        self.conv = Sequential(
            Conv2d(3, 8, kernel_size=8, stride=4), ReLU(),
            Conv2d(8, 4, kernel_size=4, stride=2), ReLU(),
            Conv2d(4, 2, kernel_size=3, stride=1), ReLU(),
        )  
        self.fc = Linear(7*7*2, 1)
        super(myConv, self).__init__()
        
    def construct(self):
        ops = [self.conv, self.fc]
        return ops
        
    def forward(self, x):
        x = self.conv(x)
        x = x.View(x.shape[0], -1)
        return self.fc(x)
      
model_trace = myConv()
print(model_trace)
for para in model_trace.parameters():
        para.data.fill_(0.001)
x = torch.rand(3, 3, 84, 84).float()
out = model_trace(x).Sum()
out.backward(show=True)

# output
Sequence(
    (0):Sequence(
        (0):Conv2d 3 -> 8
        (1):ReLU()
        (2):Conv2d 8 -> 4
        (3):ReLU()
        (4):Conv2d 4 -> 2
        (5):ReLU()
    )
    (1):Linear layer (98 X 1)
)
BackProp:
↘︎ Sum()
↘︎ Linear layer (98 X 1)
↘︎ View((3, -1))
↘︎ ReLU()
↘︎ Conv2d 4 -> 2   GRAD:torch.Size([3, 2, 7, 7]) -> torch.Size([3, 4, 9, 9])
↘︎ ReLU()
↘︎ Conv2d 8 -> 4   GRAD:torch.Size([3, 4, 9, 9]) -> torch.Size([3, 8, 20, 20])
↘︎ ReLU()
↘︎ Conv2d 3 -> 8   GRAD:torch.Size([3, 8, 20, 20]) -> torch.Size([3, 3, 84, 84])
↘︎ Input
```

see more in `torchtrace.test` and `examples`

