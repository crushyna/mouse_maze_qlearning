import json
from tensorflow.keras import backend
from datetime import datetime
from random import random, choice
from source.experience import Experience
from source.qmaze import *
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, PReLU

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


def qtrain(model, maze, epsilon, **opt):
    # global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze1 = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(qmaze1.free_cells)
    hsize = qmaze1.maze.size // 2  # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        mouse_cell = choice(qmaze1.free_cells)
        qmaze1.reset(mouse_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze1.observe()

        n_episodes = 0
        # TODO: loop starts here!
        while not game_over:
            valid_actions = qmaze1.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze1.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            # TODO: here it gets stuck!
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=2,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9: epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze1):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.now()
    dt = datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def build_model(maze: list, lr=0.001):
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
    model = build_model(maze)
    qtrain(model, maze, Constants.epsilon, epochs=1000, max_memory=8 * maze.__sizeof__(), data_size=32)
    # print("reward=", reward)
    # show(qmaze1)
