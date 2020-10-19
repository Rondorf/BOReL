
import random
import numpy as np
import metaworld
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv

ml1 = metaworld.ML1('push-v1')  # Construct the benchmark, sampling tasks


class SawyerPushEnv(SawyerReachPushPickPlaceEnv):
    def __init__(self, task={}, n_tasks=2, max_episode_steps=150, **kwargs):
        super(SawyerPushEnv, self).__init__()
        self._max_episode_steps = max_episode_steps
        self.task_type = 'push'

        self.goals = self.get_all_push_tasks()   # goals here are 6D (first 3 - obj_pos, final 3 - goal_pos)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']

    # def step(self, action):
    #     raise NotImplementedError

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()

    # def reset_model(self):
    #     raise NotImplementedError

    def reward(self, state, action):
        raise NotImplementedError

    def set_goal(self, goal):
        ''' To be discrete from self.goals '''
        raise NotImplementedError

    def sample_tasks(self, n_tasks):
        np.random.seed(1337)
        goals_idx = np.random.permutation(self.get_all_task_idx())[:n_tasks]
        tasks = [{'goal': self.goals[idx]} for idx in goals_idx]
        return tasks

    # def _get_obs(self):
    #     raise NotImplementedError

    def get_task(self):
        return super()._get_pos_goal()

    @staticmethod
    def get_all_push_tasks():
        ml1 = metaworld.ML1('push-v1')
        env = ml1.train_classes['push-v1']()
        goals = []
        for task in ml1.train_tasks:
            env.set_task(task)
            _ = env.reset()
            obj_pos = env._get_pos_objects()
            goal_pos = env._get_pos_goal()

            goal = np.hstack((obj_pos, goal_pos))
            goals.append(goal)

        return goals


# if __name__ == '__main__':
    # env = SawyerPushEnv(n_tasks=2)
    #
    # ml1 = metaworld.ML1('push-v1')  # Construct the benchmark, sampling tasks
    #
    # env = ml1.train_classes['push-v1']()  # Create an environment with task `push`
    # # task = random.choice(ml1.train_tasks)
    # # task = ml1.train_tasks[0]
    # task = ml1.train_tasks[1]
    # env.set_task(task)  # Set task
    #
    # obs = env.reset()  # Reset environment
    # a = env.action_space.sample()  # Sample an action
    # obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
