# pytorch RL in Robotics tutorial

This project is for tutorial of reinforcement learning in Robotics.
It includes examples of CartPole of Gym and Reacher of Mujoco controlled by RL.

* CartPole (OpenAI Gym)

CartPole implementation is based on the official tutorial code of pytorch ([CartPole DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)),
and some parts of the code are modified to simplify the problem.
While the original code learns from images, ours learns from simple state such like positions and velocities of CartPole using DQN.



* Reacher (Mujoco)

Mujoco simulator should be installed before you run the Reacher example. Mujoco simulator is not free, but you can freely use it for a month at first.
You can get free one-year license if you are a student and have a school e-mail(ac.kr).
This implementation also uses DQN algorithm
