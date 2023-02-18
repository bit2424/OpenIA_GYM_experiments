Here there are all the necesary depedencies to execute the project:

Pytorch base dependencies can change depending on the user here is the [link](https://pytorch.org/), here is the installation for cpu with a Linux based system:

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

pip dependencies:
    stable-baselines3
    gym
    xvfb
    ffmpeg
    xorg-dev
    libsdl2-dev
    swig
    cmake
    hydra-core

To visualize the training graphs of the testing_pipeline_tensorboard script you need to execute this command:

```
tensorboard --logdir=logs_tensorboard
```

You can also visualized the results online in the following link:

To test hiperparameter box optimization you can modify the /config/A2C.yaml file with the prefer values
and execute the following command:

```
python A2C_wandb_hydra.py -m +experiment=A2C
```