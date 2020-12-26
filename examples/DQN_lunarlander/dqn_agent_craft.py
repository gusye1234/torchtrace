import random
import numpy as np
from collections import namedtuple, deque
import torchtrace
from torchtrace import optim
import torchtrace.nn as nn
from model_craft import myDQN_network

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class Agent:
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = myDQN_network(state_size, action_size)
        self.qnetwork_target = myDQN_network(state_size, action_size)
        
        self.optimizer = optim.RMSprop(
            self.qnetwork_local.parameters(),
            lr = LR
        )
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                return self.learn(experiences, GAMMA)
        return None
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # (64, 8)
        # compute Q_target from the target network inputing next_state
        Q_target_av = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma*(Q_target_av)*(1-dones) # broadcasting works here.
        # print(Q_target.shape)
        # compute the Q_expected 
        gather = nn.Gather_last(actions.squeeze())
        Q = self.qnetwork_local(states)
        Q_expected = gather(Q)
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        #apply gradient descent
        #compute loss
        mse = nn.MSE()
        loss = mse(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward() # since we detached the Q_target, it becomes a constant and the gradients wrt Q_expected is computed only
        self.optimizer.step() # update weights
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 
        return loss.item()                    
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torchtrace.from_numpy(state).float().unsqueeze(0)
        action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torchtrace.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torchtrace.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torchtrace.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torchtrace.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torchtrace.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)