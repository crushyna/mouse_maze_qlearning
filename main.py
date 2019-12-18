from dataclasses import dataclass


@dataclass
class ProjectConstants:
    visited_mark = 0.8  # Cells visited by the mouse will be painted by gray 0.8
    mouse_mark = 0.5  # The current mouse cell will be painted by gray 0.5
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    # Actions dictionary
    actions_dict = {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down',
    }

    num_actions = len(actions_dict)

    # Exploration factor
    epsilon = 0.1


if __name__ == '__main__':
    qmaze = ProjectConstants()
