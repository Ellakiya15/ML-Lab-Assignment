import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make('PandaReach-v3', render_mode="human")
env = Monitor(env, "logs/") 

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_panda_tensorboard/")
model.learn(total_timesteps=100000)

# Save model
model.save("panda_rl_model")