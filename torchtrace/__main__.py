from .__init__ import __doc__
from rich import print
from rich.table import Column, Table

try:
    logo = """
 ______             __   __                  
/_  __/__  ________/ /  / /________ ________ 
 / / / _ \/ __/ __/ _ \/ __/ __/ _ `/ __/ -_) 
/_/  \___/_/  \__/_//_/\__/_/  \_,_/\__/\__/   
a simple auto-grad library designed for Sequential model
"""
    # font: soft
    # refer to http://patorjk.com/software/taag/#p=display&f=Soft&t=torchtrace
    print("[bold blue]"+logo+"[/bold blue]\n\n")
except:
    prYellow("Welcome to torchtrace")
    
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Author", width=10, justify="center")
table.add_column("Mail", width=26, justify="center")
table.add_column("Purpose", justify="center", style='dim')

table.add_row(
    "叶坚白",
    "jianbaiye@outlook.com",
    "for fun"
)

print(table, '\n\n\n\n')

code ="""

import torchtrace
import torchtrace.nn as nn

class Conv_craft(nn.Model):
    def __init__(self):                                                                                                          # print(Conv_craft())
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),                                          # myModel(
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),                                         #     (0):Conv(
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()).set_name("Conv")                        #         (0):Conv2d 4 -> 32        
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, 18)).set_name("Full-connect")              #         (1):ReLU() 
        super(Conv_craft, self).__init__()                                                                                       #         (2):Conv2d 32 -> 64
        self.set_name('myModel')                                                                                                 #         (3):ReLU()
                                                                                                                       →         #         (4):Conv2d 64 -> 64                          
    def construct(self):                                                                                                         #         (5):ReLU()
        return [self.conv, self.fc]                                                                                              #     )
                                                                                                                                 #     (1):Full-connect(   
    def forward(self, obs):                                                                                                      #         (0):Linear layer (3136 X 512)     
        obs = self.conv(obs)                                                                                                     #         (1):ReLU()       
        obs = obs.View(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])                                                 #         (2):Linear layer (512 X 18)
        actions = self.fc(obs)                                                                                                   #     )
        return actions                                                                                                           # )
"""

from rich.syntax import Syntax

print("[italic green]A code snippet[/italic green] ↘︎ \n\n")
print(Syntax(code, "python", theme='monokai', line_numbers=True))

print('\n\n\n')