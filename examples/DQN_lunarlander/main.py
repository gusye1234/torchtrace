import sys
sys.path.append("..")
sys.path.append("../..")

import torchtrace as torchtrace
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
seed = 0


env = gym.make('LunarLander-v2')
env.seed(seed)
torchtrace.set_seed(seed)


print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

print(env.observation_space, env.action_space)
print(env.observation_space.sample())

# from dqn_agent import Agent
from dqn_agent_craft import Agent
import time
agent = Agent(state_size=8, action_size=4, seed=0)

# watch an untrained agent running for 1 episode.


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    plt_scores_mean = []
    plt_scores_max = []
    plt_frame = []
    losses = []
    loss_frame = []
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame_num = 0
    eps = eps_start                    # initialize epsilon
    for i_episode in range(0, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            frame_num += 1
            action = agent.act(state, eps)
            #env.render()
            next_state, reward, done, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
                loss_frame.append(frame_num)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            plt_scores_mean.append(np.mean(scores_window))
            plt_scores_max.append(np.max(scores_window))
            plt_frame.append(frame_num)
            torchtrace.save('weights_list.pth', agent.qnetwork_local.seq_list()) 
            if np.mean(scores_window) >= 250. :
                np.savetxt("means.txt", plt_scores_mean)
                np.savetxt("maxs.txt", plt_scores_max)
                np.savetxt('frames.txt', plt_frame)
                np.savetxt('losses.txt', losses)
                np.savetxt('losses_frame.txt', loss_frame)
                break
            # break
    return scores, plt_scores_mean, plt_scores_max, plt_frame, losses, loss_frame

scores, means, maxs, frames, loss, loss_frame = dqn()

plt.plot(loss_frame, loss)
plt.plot(frames, means, label="mean")
plt.plot(frames, maxs, label="max")
plt.xlabel('frames')
plt.ylabel("loss")
plt.title("LunarLander-Training loss")
plt.legend()
plt.grid()
plt.savefig("lunarlander_loss.eps", dpi=600, format='eps')
plt.show()

