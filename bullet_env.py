# import for pyBullet env
import pybullet as p
import pybullet_data
import os
import glob

# robot import
from robot import pybullet_reacher as pr
import data_viz as viz
from bullet_VecNormalize import bVecNormalize


class bullet_env:
    def __init__(self, mode, env_name, max_epi_count, viz_path=None, enjoy_mode=False):     # viz_path=None은 enjoy모드
        if mode == "GUI":
            self.physicsClient = p.connect(p.GUI)  # p.GUI for graphical version
            print('connected with GUI mode')
        elif mode == "DIRECT":
            self.physicsClient = p.connect(p.DIRECT)  # p.DIRECT for non-graphical version
            print('connected with DIRECT mode')
        else:
            print('mode should be "GUI" or "DIRECT". No such mode is found')
            return -1

        # env attributes
        self.max_epi_count = max_epi_count
        self.robot_dict = {}
        self.data_viz_dict = {}
        self.viz_path = viz_path
        self.enjoy_mode = enjoy_mode


        # 로봇 등록, visualizer를 같이 등록할 것인지 여부
        # 새로운 로봇에 대해서는 여기에다 추가만 해주면 된다.
        if env_name == 'pybullet_reacher':
            self.current_env_name = 'pybullet_reacher'
            self.register_robot(pr.pybullet_reacher(self.max_epi_count), visualizer=False)
        # elif env_name == 'kuka':
        #     self.current_env_name = 'kuka'
        #     self.register_robot(pp.pan_pan(self.max_epi_count), visualizer=False)
        # TODO, additional robots can be added here.


        print('robot dict: ', self.robot_dict)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally, to use pybullet data package
        p.setGravity(0, 0, -10)

        # basic env configuration
        self.planeId = p.loadURDF('plane.urdf')


    def register_robot(self, robot, visualizer=False):
        id = robot.id

        # TODO, add condition, VecNormalize or Not
        # self.robot_dict[id] = robot
        self.robot_dict[id] = bVecNormalize(robot, ret=not self.enjoy_mode)

        if visualizer == True and self.viz_path is not None:  # visualizer 등록
            self.data_viz_dict[id] = self.init_viz(id, self.viz_path)


    def step(self, action):
        key = self.current_env_name
        obs, rew, done, info = self.robot_dict[key].step(action)


        # data viz
        if key in self.data_viz_dict:
            self.data_viz_dict[key].step_record(obs[0], rew[0], done[0], info[0])

        # TODO
        result = {self.current_env_name : (obs, rew, done, info)
                  # 여기에 다른 로봇들의 결과 추가...
                  }

        # time.sleep(1./ 240.)  # TODO
        return result


    def reset(self):
        # 모든 로봇의 위치 및 자세 등을 초기화 한다.
        # episode count를 초기화 한다.
        key = self.current_env_name
        return self.robot_dict[key].reset()


    def set_target(self, target_pos):
        key = self.current_env_name
        if key == 'pan_tilt_laserR':
            self.robot_dict[key].set_target(target_pos)


    # 폴더가 없으면 생성.. robotModel.monitor.csv
    def init_viz(self, robot_id, filepath):
        try:
            os.makedirs(filepath)
        except OSError:
            files = glob.glob(os.path.join(filepath, '*.monitor.csv'))
            for f in files:
                os.remove(f)
        return viz.data_viz(robot_id, filepath)  # env_id.filename 로 파일 이름을 생성


    def get_obs(self):
        pass

    def get_state(self):
        key = self.current_env_name
        return self.robot_dict[key].get_state()

