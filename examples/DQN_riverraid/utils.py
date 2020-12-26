import random, math
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import pickle
import logging
import torch
import torchtrace as trace
from model import ReplayBuffer
import gym
from gym import wrappers
import atari_wrapper


# --------------------------------------------------------------------------------------------------------------
def plt_result(average_100_score, max_100_score, frame_1000):
    """ Regard end of life = end of episode, and an iteration = 100 episode
    :param average_100_score: the average score of 100 episode
    :param max_100_score: the max average score of 100 episode
    :param frame_1000: frame number / 1000
    :return:
    """
    fig = plt.figure()
    plt.title(' Trainning score  ' +
              '    ' +
              ' Total iteration: ' +
              str(int(len(average_100_score))) +
              '    ')
    plt.xlabel('Steps (x 1000)')
    plt.ylabel('Score (per iteration)')
    plt.plot(frame_1000, average_100_score)
    plt.plot(frame_1000, max_100_score)
    plt.legend(["mean score/ iteration", "max score / iteration"])
    plt.grid()
    fig.savefig('Riverraid Score.eps', dpi=600, format='eps')
    fig.savefig('Riverraid Score.png')


# --------------------------------------------------------------------------------------------------------------
def plt_loss(loss_list, frame_1000):
    """ Plot the loss of every iteration
    :param loss_list: list of loss of every iteration
    :param frame_1000: frame number / 1000
    :return:
    """
    fig = plt.figure()
    plt.title(' Loss (Batch size: 32) ' +
              '    ' +
              ' Total iteration: ' +
              str(int(len(loss_list))) +
              '    ')
    plt.xlabel('Steps (x 1000)')
    plt.ylabel('Loss (per iteration) ')
    plt.plot(frame_1000, loss_list)
    plt.legend(["Loss / iteration"])
    plt.grid()
    fig.savefig('Riverraid Loss.eps', dpi=600, format='eps')
    fig.savefig('Riverraid Loss.png')



# --------------------------------------------------------------------------------------------------------------
def ReplayBuffer_Init(rep_buf_size, rep_buf_ini, env, action_space):
    replay_buffer = ReplayBuffer(rep_buf_size)
    while len(replay_buffer) < rep_buf_ini:

        observation = env.reset()
        done = False

        while not done:
            t_observation = trace.from_numpy(observation).float()
            # t_observation.shape： torch.Size([4, 84, 84])
            # t_observation.shape：torch.Size([1, 4, 84, 84])
            t_observation = t_observation.view(1, t_observation.shape[0],
                                                t_observation.shape[1],
                                                t_observation.shape[2])
            action = random.sample(range(len(action_space)), 1)[0]

            next_observation, reward, done, info = env.step(action_space[action])

            replay_buffer.push(observation, action, reward, next_observation, done)
            observation = next_observation  #

    print('Experience Replay buffer initialized')
    return replay_buffer
    pass


# --------------------------------------------------------------------------------------------------------------
def evaluate(policy_model, action_space, device, episode_true, epsilon=0.01, num_episode=10):
    env = atari_wrapper.make_atari('RiverraidNoFrameskip-v4')
    env = atari_wrapper.wrap_deepmind(env, clip_rewards=False, frame_stack=True, pytorch_img=True)
    test_scores = []
    score = 0
    episode = 0
    while episode < num_episode:

        observation = env.reset()
        done = False

        while not done:
            t_observation = trace.from_numpy(observation).float() / 255
            t_observation = t_observation.view(1, t_observation.shape[0],
                                               # t_observation.shape：torch.Size([1, 4, 84, 84])
                                               t_observation.shape[1],
                                               t_observation.shape[2])
            if random.random() > epsilon:  # choose action by epsilon-greedy
                q_value = policy_model(t_observation)
                action = q_value.argmax(1).data.cpu().numpy().astype(int)[0]
            else:
                action = random.sample(range(len(action_space)), 1)[0]

            next_observation, reward, done, info = env.step(action_space[action])
            observation = next_observation
            score += reward

        if info['ale.lives'] == 0:
            test_scores.append(score)
            episode += 1
            score = 0

    f = open("file.txt", 'a')
    f.write("%f, %d, %d\n" % (float(sum(test_scores)) / float(num_episode), episode_true, num_episode))
    f.close()

    mean_reward = float(sum(test_scores)) / float(num_episode)

    return mean_reward
