from numpy.core.multiarray import ndarray
from tensorflow.keras import backend
from source.qmaze import *
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, PReLU
from source.training import Training

maze = np.array([[1., 0., 1., 1., 1., 1., 1., 1.],
                 [1., 0., 1., 1., 1., 0., 1., 1.],
                 [1., 1., 1., 1., 0., 1., 0., 1.],
                 [1., 1., 1., 0., 1., 1., 1., 1.],
                 [1., 1., 0., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 0., 1., 0., 0., 0.],
                 [1., 1., 1., 0., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 0., 1., 1., 1.]])


def show(environment):
    plt.grid('on')
    nrows, ncols = qmaze1.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze1.maze)
    for row, col in qmaze1.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze1.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.show()
    return img


def play_game(model, qmaze, mouse_cell):
    qmaze.reset(mouse_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


def build_model(maze: ndarray, lr=0.001):
    model = Sequential()
    model.add(Dense((maze.__sizeof__()), input_shape=(64,)))
    model.add(PReLU())
    model.add(Dense(maze.__sizeof__()))
    model.add(PReLU())
    model.add(Dense(Constants.num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    const1 = Constants()
    '''
    canvas, reward, game_over = qmaze1.act(Constants.DOWN)
    canvas, reward, game_over = qmaze1.act(Constants.DOWN)
    canvas, reward, game_over = qmaze1.act(Constants.RIGHT)
    '''
    # qmaze1 = Qmaze(maze)
    # show(qmaze1)
    # experience1 = Experience()
    model1 = build_model(maze)
    Training.qtrain(model1, maze, Constants.epsilon, epochs=1000, max_memory=8 * maze.__sizeof__(), data_size=32)
    # print("reward=", reward)
    # show(qmaze1)
