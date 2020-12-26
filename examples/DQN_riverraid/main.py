import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
import torch
import torchtrace
import torchtrace as trace
import torchtrace.nn as nn
import torchtrace.optim as optim
from model import DQN_craft, ReplayBuffer
import random
import gym
from gym import wrappers
import atari_wrapper
import math
import pickle
import logging
from collections import deque
from utils import plt_result, plt_loss
from utils import ReplayBuffer_Init, evaluate

seed = 0
trace.set_seed(0)
# --------------------------------------------------------------------------------------------------------------
# Hyperparameters :
lr = 0.0000625  # Learning rate
alpha = 0.95  # For RMSprop momentum
max_episodes = 40100 # About 2500000 frame
batch_size = 32
target_update = 10000  # Not use now
gamma = 0.99
# rep_buf_size = 1000000
rep_buf_size = 1000 # for test
# rep_buf_ini = 50000
rep_buf_ini = 50 # for test
skip_frame = 4
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 300000
Soft_update = True
TAU = 2e-3  # For soft update
Evaluation = True  # Determine whether evaluate the agent during training
evaluate_frequency = 400  # Frequency of evaluate
evaluate_episodes = 15  # Episodes of evaluation
evaluate_epsilon = 0.01  # epsilon of evaluation
save_frequency = 100  # Frequency of save model
test_stander = 7200  # If test score > test_stander, stop train


# --------------------------------------------------------------------------------------------------------------

def epsilon_by_frame(step_idx):
    """Epsilon Decay: From 1 to 0.01 ：In 1M Frames"""
    epsilon_true = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_idx / epsilon_decay)
    return epsilon_true


# --------------------------------------------------------------------------------------------------------------

def main():

    # Initialize environment and :
    env = atari_wrapper.make_atari('RiverraidNoFrameskip-v4')
    env = atari_wrapper.wrap_deepmind(env, clip_rewards=True, frame_stack=True, pytorch_img=True)
    action_space = [a for a in range(env.action_space.n)]

    # Initialize DQN Model and optimizer:
    policy_model = DQN_craft()
    target_model = DQN_craft()
    print(policy_model)
    target_model.eval()
    target_model.load_seq_list(target_model.seq_list())
    optimizer = optim.RMSprop(policy_model.parameters(), lr=lr, alpha=alpha)
    
    # -------------------------------------------------
    # Initialize the Replay Buffer
    replay_buffer = ReplayBuffer_Init(rep_buf_size, rep_buf_ini, env, action_space)

    # Use log to record the performance
    logger = logging.getLogger('dqn_Riverraid')
    logger.setLevel(logging.INFO)
    logger_handler = logging.FileHandler('./dqn_Riverraid.log')
    logger.addHandler(logger_handler)

    # --------------------------------------------------------------------------------------------------------------
    # Training part, Initialization below
    env.reset()
    score = 0
    episode_scores = []  # A list to record all episode_true score
    episode_true = 0  # we regard end of life = end of episode, since there are 4 lives in RiverRaid, thus,
    # one episode_ture = 4 episodes
    num_frames = 0
    episode = 0
    average_100_episode = []  # For plot
    max_100_episode = []  # For plot
    frame_1000 = []  # For plot
    last_25episode_score = deque(maxlen=25)  # for plot
    loss_list = []  # for plot
    loss_running = 0  # for log
    # End of initialization
    # --------------------------------------------------------------------------------------------------------------

    while episode < max_episodes:

        observation = env.reset()
        done = False

        while not done:

            t_observation = torch.from_numpy(observation).float() / 255
            # t_observation = t_observation.view(1, t_observation.shape[0],
            #                                     t_observation.shape[1],
            #                                     t_observation.shape[
            #                                         2])  # t_observation.shape：torch.Size([1, 4, 84, 84])
            t_observation = t_observation.unsqueeze(0)
            epsilon = epsilon_by_frame(num_frames)
            if random.random() > epsilon:  # choose action by epsilon-greedy
                q_value = policy_model(t_observation)
                action = q_value.argmax(1).data.numpy().astype(int)[0]
            else:
                action = random.sample(range(len(action_space)), 1)[0]

            next_observation, reward, done, info = env.step(action_space[action])
            replay_buffer.push(observation, action, reward, next_observation, done)
            observation = next_observation
            num_frames += 1  # update frame
            score += reward

            # Update policy
            if len(replay_buffer) > batch_size and num_frames % skip_frame == 0:
                observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)

                observations = trace.from_numpy(np.array(observations) / 255).float()

                actions = trace.from_numpy(np.array(actions).astype(int)).float()
                actions = actions.view(actions.shape[0], 1)  # torch.Size([32, 1])

                rewards = trace.from_numpy(np.array(rewards)).float()
                rewards = rewards.view(rewards.shape[0], 1)  # torch.Size([32, 1])

                next_observations = trace.from_numpy(np.array(next_observations) / 255).float()

                dones = trace.from_numpy(np.array(dones).astype(int)).float()
                dones = dones.view(dones.shape[0], 1)  # torch.Size([32, 1])

                q_values = policy_model(observations)  # torch.Size([32, 18])
                next_q_values = target_model(next_observations)  # torch.Size([32, 18])

                q_value = q_values.Gather(actions.squeeze().long())  # torch.Size([32, 1])
                next_q_value = next_q_values.max(1)[0].unsqueeze(1)  # torch.Size([32, 1])
                expected_q_value = rewards + gamma * next_q_value * (1 - dones)

                mse = nn.MSE()
                loss = mse(q_value, expected_q_value)
                # loss = huber_loss(q_value, expected_q_value)
                loss_running = loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"\rloss: {loss}", end='')
                # Soft update or Hard update (common dqn)
                if Soft_update:
                    for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
                        target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
                else:
                    if num_frames % target_update == 0:
                        # target_model.load_state_dict(policy_model.state_dict())
                        target_model.load_seq_list(target_model.seq_list())

        episode += 1
        # End of episode_true, reset the score
        if info['ale.lives'] == 0:
            episode_scores.append(score)
            last_25episode_score.append(score)

            # record in the log
            logger.info('Frame: ' + str(num_frames) +
                        '  / Episode: ' + str(episode_true) +
                        '  / Average Score : ' + str(int(score)) +
                        '  / epsilon: ' + str(float(epsilon)) +
                        '  / loss ： ' + str(float(loss_running)))
            pickle.dump(episode_scores, open('./dqn_Riverraid_mean_scores.pickle', 'wb'))

            episode_true += 1
            score = 0

            print('\r Episode_true {} \t Average Score(last 100 episodes) {:.2f} '.
                  format(episode_true, np.mean(last_25episode_score)),
                  end=" ")
            # Update the log
            if episode_true % 25 == 1:
                logger.info(
                    'Frame: ' + str(num_frames) +
                    '  / Episode: ' + str(episode_true) +
                    '  / Average Score : ' + '         ' +
                    '  / epsilon: ' + str(float(epsilon)) +
                    '  / last_100episode_score: ' + str(float(np.mean(last_25episode_score))))

                print("episode_ture: ", episode_true, "average_100_episode :", np.mean(last_25episode_score))
                print("episode:", episode)
            # This "if " is for plot (to store the data of per iteration )
            if episode_true % 25 == 0:
                average_100_episode.append(np.mean(last_25episode_score))
                max_100_episode.append(np.max(last_25episode_score))
                loss_list.append(loss_running)
                frame_1000.append(num_frames / 1000.)
            # plot the scores and loss and save the picture
            if episode_true % 500 == 0:
                plt_result(average_100_episode, max_100_episode, frame_1000)
                plt_loss(loss_list, frame_1000)
            # Evaluation Part
            if Evaluation:
                if episode_true % evaluate_frequency == 0:
                    test_score = evaluate(policy_model,
                                          action_space,
                                          device,
                                          episode_true,
                                          epsilon=evaluate_epsilon,
                                          num_episode=evaluate_episodes)
                    print("test_score : ", test_score, "  ", "test episodes: ", evaluate_episodes)

                    if test_score > test_stander:  # Save the model if the test score > test_stander
                        trace.save('./dqn_RiverRaid_policy_model_state_dict.pth',policy_model.seq_list())
                        print("Test score > %d , stop train" % test_stander)
                        break
        # Save the model
        if episode % save_frequency == 0:
            trace.save('./dqn_RiverRaid_policy_model_state_dict.pth',policy_model.seq_list())

    plt_result(average_100_episode, max_100_episode, frame_1000)
    plt_loss(loss_list, frame_1000)
    pass


# --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
