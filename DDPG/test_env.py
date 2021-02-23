# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:agent.py
# software: PyCharm

from continuous_cart_pole_env import ContinuousCartPoleEnv

# env = gym.make(id='CarRacing-v0')
env = ContinuousCartPoleEnv()
env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print(env.action_space)
