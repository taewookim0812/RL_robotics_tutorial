import pybullet as p
import numpy as np
import random
import gym
import os
import time

from vecEnv import VecEnv


class pybullet_reacher(VecEnv):
    def __init__(self, max_epi_count, observation_space=None, action_space=None):
        VecEnv.__init__(self,
                        observation_space=gym.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=(11,)),
                        action_space=gym.spaces.box.Box(low=-1, high=1, shape=(2,))
                        )
        # robot attribute
        self.id = 'pybullet_reacher'
        self.max_epi_count = max_epi_count
        self.epi_count = 0
        self.num_frame_skip = 2

        # parameter value
        self.endEffectorIndex = 2
        self.numJoints = 2
        self.convergence_thres = 0.02  # 0.01m = 10cm
        self.scale_const = 240   # pybullet은 1입력 -> 0.238732(deg) or 0.004166(rad) /step, 따라서 1/0.004166 = 240

        # self.observation_space = gym.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=(11,))  # TODO
        # self.action_space = gym.spaces.box.Box(low=-1, high=1, shape=(self.numJoints,))

        self.robotStartPos = [0, 0, 0]  # 나중에 가우시안 노이즈 줘서?
        self.robotStartOrientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        # current working directory
        self.cwdir = os.getcwd()

        pos = np.random.uniform(low=-.5, high=.5, size=2)
        pos = np.append(pos, 0.025)
        self.targetId = p.loadURDF(self.cwdir + '/urdf/target_point.urdf', pos, useFixedBase=True)

        self.robotId = p.loadURDF(self.cwdir + '/urdf/pybullet_reacher_robot.urdf', self.robotStartPos,
                                  self.robotStartOrientation,
                                  useFixedBase=True)


    # return observation as [cos(j1pos), sin(j1pos), cos(j2pos), sin(j2pos), j1vel, j2vel, (tx, ty), (dx, dy, dz)]
    # 1 x 11
    def get_obs(self):
        observation = []
        # joint position jv[0]
        jp1 = p.getJointState(self.robotId, 0)  # pan1 joint
        jp2 = p.getJointState(self.robotId, 1)  # pan2 joint

        # cos, sin of pan1, 2 angle
        observation.append(np.cos(jp1[0]))
        observation.append(np.cos(jp2[0]))

        observation.append(np.sin(jp1[0]))
        observation.append(np.sin(jp2[0]))

        # observation.append(jp1[0] % (2 * np.pi))
        # observation.append(jp2[0] % (2 * np.pi))

        # target position
        tp = p.getLinkState(self.targetId, 0)
        observation += [tp[4][i] for i in range(len(tp[4]) - 1)]  # only includes x, y term

        # joint velocity jv[1]
        for j in range(self.numJoints):
            jv = p.getJointState(self.robotId, j)
            observation.append(jv[1] / self.scale_const * self.num_frame_skip)  # 속도는 frame skip 한 만큼 곱해줄 것


        # pan2 (finger point)
        fp = p.getLinkState(self.robotId, self.endEffectorIndex)  # laser pointer, lp[4] is worldLinkPosition
        # observation += [ep[4][i] for i in range(len(ep[4])-1)]      # from tuple to list and concat, only x, y term

        dp = np.array(fp[4]) - np.array(tp[4])  # laserPos - targetPos, ndarray
        observation += dp.tolist()
        # temp += [tp[4][i] for i in range(len(tp[4]))]      # from tuple to list and concat

        return np.array([observation])  # return as a list


    def get_reward(self, action):
        # TODO, delayed reward..
        # reward, dist [target - laser-end]
        fp = p.getLinkState(self.robotId, self.endEffectorIndex)  # finger tip, fp[4] is worldLinkPosition
        tp = p.getLinkState(self.targetId, 0)
        distErr = -np.linalg.norm(np.subtract(fp[4], tp[4]))
        ctrlErr = -np.square(action).sum()
        # print('ctrlErr: ', ctrlErr)
        reward = distErr + ctrlErr  # TODO
        return distErr, ctrlErr, reward


    def step(self, action):
        if len(action) != self.numJoints:
            print('action dim is not matched!')
            return -1

        const = 240     # TODO, 240
        # print('action: ', action * const)

        distErr, ctrlErr, reward = self.get_reward(action)


        desq = np.zeros(self.numJoints)
        for j in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.robotId, jointIndex=j, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[j] * self.scale_const, velocityGain=1, force=100)

        # frame skip에 따른 누적된 reward 계산
        for i in range(self.num_frame_skip):
            p.stepSimulation()
            dErr, cErr, rew = self.get_reward(action)
            distErr += dErr
            ctrlErr += cErr
            reward += rew


        # iterable 해야 하기 때문에 []로 처리해준다.
        # observation
        obs = self.get_obs()

        # Done, frame skip에서 마지막 step을 기준으로 Done 여부를 설정한다.
        # if np.abs(distErr) < self.convergence_thres:
        #     done = True
        # else:
        #     done = False

        done = False

        # Info.
        info = [dict(reward_dist=distErr, reward_ctrl=ctrlErr)]

        self.epi_count += 1

        # End of Episode
        if self.epi_count >= self.max_epi_count or done == True:
            self.epi_count = 0
            obs = self.reset()
            done = True

        return obs, [reward], np.array([done]), info


    def reset(self):
        # retarget
        while True:
            goal = np.random.uniform(low=-.4, high=.4, size=2)
            if np.linalg.norm(goal) < 2:
                break
        goal = np.append(goal, 0.025)   # plane 위쪽으로 target이 생성되도록 z축 값을 offset
        p.resetBasePositionAndOrientation(bodyUniqueId=self.targetId, posObj=goal, ornObj=self.robotStartOrientation)

        # reset robot position, velocity
        for j in range(self.numJoints):
            random_pos = np.random.uniform(low=-0.1, high=0.1, size=1)
            random_vel = np.random.uniform(low=-0.05, high=0.05, size=1)
            p.resetJointState(bodyUniqueId=self.robotId, jointIndex=j, targetValue=random_pos, targetVelocity=random_vel)

        p.stepSimulation()

        # print('env reset!')
        return self.get_obs()























