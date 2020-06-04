from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import itertools as itt


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
        max_steps=None,
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
                # reward += 10
                reward += 0.1*self.max_steps
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


class CrossingEnvMod(MiniGridEnvMod):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None, agent_start_pos=(1,1),
        agent_start_dir=0):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=1000,  # 4*size*size
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def set_initial_pos_dir(self, start_pos, start_dir):
        self.agent_start_pos = start_pos
        self.agent_start_dir = start_dir

    def set_max_steps(self, max_steps):
        super().__init__(
            grid_size=self.width,
            max_steps=max_steps,  # 4*size*size
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=None
        )

class LavaCrossingEnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=1)

class LavaCrossingS9N2EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=2)

class LavaCrossingS9N3EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=3)

class LavaCrossingS11N5EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=11, num_crossings=5)

register(
    id='MiniGrid-LavaCrossingModS9N1-v0',
    entry_point='gym_minigrid.envs:LavaCrossingEnvMod'
)

register(
    id='MiniGrid-LavaCrossingModS9N2-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N2EnvMod'
)

register(
    id='MiniGrid-LavaCrossingModS9N3-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N3EnvMod'
)

register(
    id='MiniGrid-LavaCrossingModS11N5-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS11N5EnvMod'
)

class SimpleCrossingEnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=1, obstacle_type=Wall)

class SimpleCrossingS9N2EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=2, obstacle_type=Wall)

class SimpleCrossingS9N3EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=9, num_crossings=3, obstacle_type=Wall)

class SimpleCrossingS11N5EnvMod(CrossingEnvMod):
    def __init__(self):
        super().__init__(size=11, num_crossings=5, obstacle_type=Wall)

register(
    id='MiniGrid-SimpleCrossingModS9N1-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingEnvMod'
)

register(
    id='MiniGrid-SimpleCrossingModS9N2-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N2EnvMod'
)

register(
    id='MiniGrid-SimpleCrossingModS9N3-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N3EnvMod'
)

register(
    id='MiniGrid-SimpleCrossingModS11N5-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS11N5EnvMod'
)
