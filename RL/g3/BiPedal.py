import gym
import Box2D
env = gym.make('BipedalWalker-v2')
obs = env.reset()
print(env.action_space.high)
print(env.observation_space.high)
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample())
