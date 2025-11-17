from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from enum import Enum 

@dataclass
class Pos:
    x: int
    y: int

    def to_tuple(self):
        return (self.x, self.y)

class Action(str, Enum):
    up = "UP"
    left = "LEFT"
    down = "DOWN"
    right = "RIGHT"

action_do_delta = {
    Action.up: (-1, 0),
    Action.down: (1, 0),
    Action.left: (0, -1),
    Action.right: (0, 1)
}

class GridWorld:
    def __init__(
        self, 
        grid_size: int, 
        start_pos: Pos, 
        end_pos: Pos,
        obstacles: List[Pos],
        rewards: Tuple[int, int, int]
    ) -> None:
        self.grid_size = grid_size
        self.world = np.zeros((grid_size, grid_size))
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.obstacles = obstacles
        rows, cols = zip(*[obstacle.to_tuple() for obstacle in obstacles])
        self.world[rows, cols] = 1 # obstacles
        self.world[end_pos.x, end_pos.y] = 2 # end goal 
        self.rewards = rewards
        self.step_reward, self.obstacle_reward, self.goal_reward = self.rewards
        self.current_pos = start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: str):
        dx, dy = action_do_delta[Action[action]]
        new_x = self.current_pos.x + dx
        new_y = self.current_pos.y + dy

        is_done = False
        if not(new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size):
            self.current_pos = Pos(x=new_x, y=new_y)
        else:
            if new_x == self.end_pos.x and new_y == self.end_pos.y:
                is_done = True

        step_reward = self.rewards[self.world[self.current_pos.x, self.current_pos.y]]
        return self.current_pos, step_reward, is_done
