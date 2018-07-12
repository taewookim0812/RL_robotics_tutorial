# pytorch RL tutorial for Robotics

* Youtube:
(https://www.youtube.com/watch?v=gWaiMGr4Le8&t=2s)

<br />


This is a pytorch implementation of Reinforcement Learning for Robotics.
Based on virtual simulation environments such as OpenAI Gym, Mujoco and pybullet,
We implemented some examples of reinforcement learning for robotics using simple robots.
Our article will be contributed to the July issue of 2018 of the "Human and Robot" of KROS
and can be found
[here](https://github.com/gd-goblin/RL_robotics_tutorial/tree/master/document).


## CartPole (OpenAI Gym)

<p align="center">
    <img width="640" height="480" src=img/cartpole.png>
</p>

Our CartPole implementation is based on the official tutorial code of pytorch ([CartPole DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)),
and some parts of the code are modified to simplify the problem.
While the original code learns from images, ours learns from simple state such like positions and velocities of CartPole using DQN.

* Training:
cartpole_train.py

* Evaluation:
cartpole_eval.py

<br />
<br />

## Reacher (Mujoco)

<p align="center">
    <img width="640" height="480" src=img/Reacher(Mujoco).png>
</p>

Mujoco simulator should be installed before you run the Reacher example. Mujoco simulator is not free, but you can freely use it for the first month.
You can get free one-year license if you are a student and have a school e-mail(ac.kr).
This implementation also uses DQN algorithm. PPO version can be found [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

* Training:
reacher_train.py

* Evaluation:
reacher_eval.py

<br />
<br />

## Reacher (pybullet)

<p align="center">
    <img width="640" height="480" src=img/Reacher(pybullet).png>
</p>

Based on ikostrikov's implementation(https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
we organized pybullet simulation environment with Reacher robot same as Mujoco's using urdf format.

* Training:
pybullet_train_main.py

* Evaluation:
pybullet_enjoy.py


