import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import wandb
from datetime import datetime
import os

now = datetime.now()
now_str = now.strftime("%d_%m_%Y %H:%M:%S")


models_base_dir = "models/"
logdir = "logs_tensorboard/"+now_str


models = ["A2C","PPO"]

for model_name in models:
    if not os.path.exists(models_base_dir+model_name):
        os.makedirs(models_base_dir+model_name)

if not os.path.exists(logdir):
    os.makedirs(logdir)

TIMESTEPS = 1000

for model_name in models:
    env = gym.make('LunarLander-v2')
    env.reset()
    model = None
    if(model_name == "A2C"):
        model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=logdir)
    if(model_name == "PPO"):
        model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=logdir)

    iters = 0
    for i in range(30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
        if(i == 15):
            model.save(f"{models_base_dir+model_name}/{TIMESTEPS*i}")