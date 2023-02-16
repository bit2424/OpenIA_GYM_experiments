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

To visualize the training graphs of the testing_pipeline_tensorboard script you need to execute this command:

```
tensorboard --logdir=logs_tensorboard
```