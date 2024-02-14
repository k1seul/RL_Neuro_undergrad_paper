import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
import itertools
from pygame_screen_recorder import pygame_screen_recorder as pgr


class GridWorld(gym.Env):
    """Discreate version of Monkey 3D maze, has 2D discreate state space with 4 actions left, right, up, down"""

    version_num = "1.6.1"
    """
    version update: added action mask to the 2D enviornment 
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, render_mode=None, no_reward=False, file_name=None):
        """change render_mode to see pygame window
        no_reward is for making the env with no reward for latent learning(only negative reward exists when hitting the wall)
        file_name is for directory input to save the .gif file of pygame gameplay
        """
        self.window_size = 800
        self.size = 3  # 11 x 11 grid world
        self.no_reward = no_reward

        # minimal_state_n is given as (x,y) coordinate
        self.state_n = 2
        # action is four arrows with tank control
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        # 0:go_front 1: turn_left, 2: turn_right ;; 3: go_behind(can be turned off)(off by default)
        self.action_n = 4
        self.max_episode_step = 2000


        ## change the monkey location to veiw monkey as brown square
        self._monkey_location = None

        ## front_sight is defined as 0: no_wall, 1: wall, 2: small_reward, 3: jackpot_reward

        self.observation_space = spaces.Dict(
            {
                "agent_coordinate": spaces.Box(
                    low=0, high=11, shape=([2]), dtype=np.int64
                ),
            }
        )
        self.action_space = spaces.Discrete(self.action_n)
        self.init_map()

        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.recrdr = pgr(file_name)  # init recorder object

    def init_map(self):



             
              

        self.whole_map = np.ones([self.size, self.size])

    


        self.reward_location = np.array(
            [[2,2]]
        )

    def init_reward(self, reward=None, start=None):
        """
        initialize goal location of current trial if reward, start idx aren't given
         it will be randomized
         """
        self.trial_goal = np.array([2,2])
        self.trial_start = np.array([0,0])

    def _get_obs(self):
        """
        observation without any memory inputed back to the state
        default observation will be agent coordinate
        """
        observation = {
            "agent_coordinate": self._agent_location,
        }

        obs_array = np.concatenate([observation[key] for key in observation])

        return obs_array

    def _get_info(self, state):
        """Return action mask as info, action mask restrict the action q-values to >0 and zero value 
        possible action from state is given as 1 and impossible action as 0
        mutiply this to the q-values outcome of dqn for action mask implementation 
        """
        info = np.zeros([4]) 


        return info 

    def reset(self, seed=None, reward=None, start=None):
        super().reset(seed=seed)
        self.init_reward()

        ## initializing starting agent state
        self._agent_location = self.trial_start
        observation = self._get_obs()
        info = self._get_info(self._agent_location)
        self.step_count = 1

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.step_count += 1
        truncated = False
        # 0:go_front 1: turn_left, 2: turn_right ;; 3: go_behind(can be turned off)(off by default)
        if action == 0 or action == 1 or action == 2 or action == 3:
            new_location = self._agent_location + self._action_to_direction[action]
        else:
            raise ValueError("action is not defined")

        
        new_location = np.clip(new_location, 0, self.size - 1)

        wall_hit = np.array_equal(self._agent_location, new_location)

        self._agent_location = new_location

        terminated = np.array_equal(self._agent_location, self.trial_goal)
        if self.step_count >= self.max_episode_step:
            truncated = True


        observation = self._get_obs()
        info = self._get_info(self._agent_location)

        if terminated:
            reward = 8 if not (self.no_reward) else 0
        elif wall_hit:
            reward = - 0.1 
        else:
            reward = 0

        if self.no_reward:
            terminated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.trial_goal,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
       
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )


        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.recrdr.click(self.window)

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


"""
if __name__ == "__main__":
    from MonkeyMazeEnv import MonkeyMazeEnv 
from MonkeyPath import MonkeyPath
import pygame 
import pickle
import numpy as np 
### For Playing the env with monkey!!! select the monkey name "p" or "s" and play within the env with them!
terminated = False
truncated = False 
env = GridWorld(render_mode="human", no_reward=False)




trajectories = [] 
state, info = env.reset()
terminated = False
truncated = False 
trajectories.append(state) 

pygame.event.clear()
sum_reward = 0
trial_len = 0 

while not(terminated or truncated):
    

    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_UP:
                action = 3
            elif ev.key == pygame.K_LEFT:
                action = 2
            elif ev.key == pygame.K_RIGHT:
                action = 0
            elif ev.key == pygame.K_DOWN:
                action = 1

            obs, reward, done, truncated, info = env.step(action)

            print(action) 
            trial_len += 1 

            print(f"obs: {obs}, reward:{reward}")
            sum_reward = sum_reward + reward

            trajectories.append(obs) 

    
            if done or truncated:
                print("Game over! Final score: {}".format(sum_reward))
                terminated = done 
                with open("example_trajectories", 'wb') as fp:
                    pickle.dump(trajectories, fp)
                    print("trajectories saved!")

                break 
                
"""

