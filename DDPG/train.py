# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm

from continuous_cart_pole_env import ContinuousCartPoleEnv
from agent import Agent
import numpy as np


def train_episode(env, agent, noise, reward_scale):
    obs = np.array([env.reset()])
    scores = 0
    steps = 0

    while True:
        steps += 1
        action = agent.actor_change(obs).numpy()[0][0]
        # TODO:add noise
        action = np.clip(np.random.normal(action, noise), -1.0, 1.0)
        next_state, reward, done, _ = env.step(action)
        scores += reward
        agent.store_transition(obs[0], action, next_state, reward_scale * reward, done)
        obs = np.array([next_state])

        # if agent.replay_buffer.cntr > MEMORY_WARMUP_SIZE:
        if agent.replay_buffer.cntr > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            agent.learn()

        if done or steps >= 200:
            break

    return scores


def evaluate(env, agent, render=False):
    eval_reward = []

    for j in range(5):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            state = np.expand_dims(state, axis=0)
            action = agent.actor_change(state).numpy()[0][0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
            if steps >= 200 or done:
                break

        eval_reward.append(total_reward)

    return np.mean(eval_reward)


if __name__ == '__main__':

    ACTOR_LR = 1e-3  # Actor model (learning rate)
    CRITIC_LR = 1e-3  # Critic model (learning rate)

    GAMMA = 0.99  # decay factor of reward
    TAU = 0.001  # soft updating
    MEMORY_SIZE = int(1e6)  # replay memory size
    MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # warm up
    BATCH_SIZE = 128
    REWARD_SCALE = 0.1  # scaling factor of reward
    NOISE = 0.05  # the variance of action

    TRAIN_EPISODE = 6000  # one episode = one game

    # create env
    env = ContinuousCartPoleEnv()
    obs_dim = env.observation_space.shape
    # print(obs_dim)
    action_dim = env.action_space.shape
    # print(action_dim)

    agent = Agent(obs_dim, action_dim,
                  BATCH_SIZE, ACTOR_LR, CRITIC_LR,
                  100, len(action_dim), 100, 1,
                  max_size=MEMORY_SIZE, gamma=GAMMA, tau=TAU)

    while agent.replay_buffer.cntr <= MEMORY_WARMUP_SIZE:
        train_episode(env, agent, NOISE, REWARD_SCALE)

    for i in range(TRAIN_EPISODE):
        train_episode(env, agent, NOISE, REWARD_SCALE)

        if i % 50 == 0 and i > 0:
            scores = evaluate(env, agent)
            print('episode:{} --- scores:{}'.format(i, scores))
