import gym
import timeit
from stable_baselines3 import PPO
import wandb
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

seed = 1
lr = 0.001

models_base_dir = "models/PPO"+str(seed)
logdir = "logs_wandb/"

def main():
    wandb.tensorboard.patch(root_logdir=logdir)
    wandb.init(project='team-forest-dashboard', sync_tensorboard=True)
    wandb.run.name = "PPO_"+str(seed)


    if not os.path.exists(models_base_dir):
        os.makedirs(models_base_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    TIMESTEPS = 500

    env = gym.make('LunarLander-v2')
    env.reset()

    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=logdir, seed=seed,device='cuda',learning_rate=0.01)


    iters = 0
    for i in range(30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= "PPO")
        if(i == 5):
            model.save(f"{models_base_dir}/{TIMESTEPS*i}")

    wandb.finish()

t_0 = timeit.default_timer()
main()
t_1 = timeit.default_timer()
elapsed_time = round((t_1 - t_0), 3)
print(f"Elapsed time: {elapsed_time}")