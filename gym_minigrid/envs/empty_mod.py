from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MiniGridEnvMod(MiniGridEnv):

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnvMod.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)  # not used

        self.current_reward = 0
        self.accum_reward = 0

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        #return 1 - 0.9 * (self.step_count / self.max_steps)
        return 10 - 0.1 * self.step_count

    def step(self, action):
        self.step_count += 1

        reward = -0.1
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                # reward = self._reward()
                reward += 10
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True
            # reward = self._reward() # mod

        obs = self.gen_obs()

        self.current_step_reward = reward
        self.accum_reward += reward

        return obs, reward, done, {}

class EmptyEnvMod(MiniGridEnvMod):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def set_initial_pos_dir(self, start_pos, start_dir):
        self.agent_start_pos = start_pos
        self.agent_start_dir = start_dir

    def set_max_steps(self, max_steps):
        super().__init__(
            grid_size=self.width,
            max_steps=max_steps,  # 4*size*size
            # Set this to True for maximum speed
            see_through_walls=True,
        )


class EmptyEnvMod5x5(EmptyEnvMod):
    def __init__(self):
        super().__init__(size=5)

class EmptyRandomEnvMod5x5(EmptyEnvMod):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnvMod6x6(EmptyEnvMod):
    def __init__(self):
        super().__init__(size=6)

class EmptyRandomEnvMod6x6(EmptyEnvMod):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnvMod16x16(EmptyEnvMod):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-EmptyMod-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnvMod5x5'
)

register(
    id='MiniGrid-EmptyMod-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnvMod5x5'
)

register(
    id='MiniGrid-EmptyMod-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnvMod6x6'
)

register(
    id='MiniGrid-EmptyMod-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnvMod6x6'
)

register(
    id='MiniGrid-EmptyMod-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnvMod'
)

register(
    id='MiniGrid-EmptyMod-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnvMod16x16'
)
