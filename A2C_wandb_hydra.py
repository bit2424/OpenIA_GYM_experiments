import timeit
import gym
from stable_baselines3 import A2C
import wandb
import os
import hydra

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="A2C.yaml")
def main(cfg: DictConfig):
    
    seed = cfg['seed']
    lr = float(cfg['lr'])

    models_base_dir = "models/A2C_"+str(seed)
    logdir = "logs_wandb/"
    
    if(cfg['process_id'] == 0):
        wandb.tensorboard.patch(root_logdir=logdir)
    
    wandb.init(project='team-forest-dashboard', sync_tensorboard=True)
    wandb.run.name = "A2C_seed:"+str(seed)+"_lr:"+str(lr)


    if not os.path.exists(models_base_dir):
        os.makedirs(models_base_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    TIMESTEPS = 500

    #Env
    env = gym.make('LunarLander-v2')
    env.reset()

    model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=logdir, seed=seed,device='cuda',learning_rate=lr)

    
    for i in range(30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= "A2C")
        if(i == 5):
            model.save(f"{models_base_dir}/{TIMESTEPS*i}")

    wandb.finish()

t_0 = timeit.default_timer()
main()
t_1 = timeit.default_timer()
elapsed_time = round((t_1 - t_0), 3)
print(f"Elapsed time: {elapsed_time}")