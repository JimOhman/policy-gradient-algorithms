import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, action_space, state_space):
        super(SimpleNet, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, action_space)


    def forward(self, x):
    	x = F.relu(self.fc1(x))
    	return self.fc2(x)


class SimpleNetWithValueHead(nn.Module):
    def __init__(self, action_space, state_space):
        super(SimpleNet, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        
        self.fc1 = nn.Linear(state_space, 128)
        self.value = nn.Linear(128, 1)
        self.logits = nn.Linear(128, action_space)


    def forward(self, x):
    	x = F.relu(self.fc1(x))
    	return self.logits(x), self.value(x)
