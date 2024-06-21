import numpy as np
import torch
from torch import nn
from tictactoe import Board, Game
from random_agent import RandomAgent
import matplotlib.pyplot as plt
from copy import deepcopy

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class TicTacToeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # x = torch.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        # x = self.fc2(x)              # Apply the second linear transformation
        return logits


model = TicTacToeNN().to(device)
loss_fn = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action


def train_data(data, model):
    num_epochs = 50

    for epoch in range(num_epochs):
        # output = []
        for squares, q in data:
            model.train()

            predicted_q = model(squares)
            target_q = q

            loss = criterion(predicted_q, target_q)  # find out why this is working.
            # Zero the gradients
            optimizer.zero_grad()

            # # Forward pass
            # output.append(model(squares))

            # # Compute the loss
            # loss = criterion(output, q)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

    # model.train()
    # loss = loss_fn(torch.tensor(predictive_qs), torch.tensor(target_qs))
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()


Q = {}
rewards = list()
gamma = 0.95
alpha = 0.95


def board_conversion_to_2x9(squares):
    x = []
    o = []

    for i in range(len(squares)):
        for j in range(len(squares[i])):
            if squares[i][j] == 1:
                x.append(1)
                o.append(0)
            elif squares[i][j] == 2:
                x.append(0)
                o.append(1)
            else:
                x.append(0)
                o.append(0)

    return torch.tensor(x + o).float()


def nn_play(game, model):
    possible_states = list()
    # Make hashes of each possible next game state
    for x in range(3):
        for y in range(3):
            if game.board.squares[x][y] == 0:
                game_copy = deepcopy(game)
                game_copy.play(x, y)
                possible_states.append((x, y, game_copy.board.squares))
    # Get q_values of all the possible next game states from Q table
    # or initialized with 0.5 if the state was never seen before
    q_values = [
        model(board_conversion_to_2x9(possible_state[2]).cuda()).detach().cpu().numpy()
        for possible_state in possible_states
    ]
    # Use epsilon greedy to make a move
    picked_value = epsilon_greedy_policy(q_values, 0.1)
    x, y, _ = possible_states[picked_value]
    return (x, y, q_values[picked_value])


def extract_max_possible_q(squares, model):
    # Setting up the game from the given state
    game = Game()
    game.board.squares = squares
    game.turn = np.sum(game.board.squares != 0)
    # Explore possible next states
    max_Q = 0
    for x in range(3):
        for y in range(3):
            if game.board.squares[x][y] == 0:
                game_copy = deepcopy(game)
                game_copy.play(x, y)
                Q_possible_state = model(board_conversion_to_2x9(squares).cuda())
                if Q_possible_state > max_Q:
                    max_Q = Q_possible_state
    return max_Q


training_data = []
rewards = []

win_perc = []

for i in range(1001):
    random_agent = RandomAgent()
    visited_states = list()
    game = Game()
    while not game.done:
        x, y, Q = nn_play(game, model)
        game.play(x, y)
        visited_state = deepcopy(game.board.squares)
        visited_states.append((visited_state, Q))
        if game.done:
            break
        random_agent.play(game)
    # Keep track of q-agent's rewards for our own monitoring
    rewards.append(game.p1_reward)
    # Update Q table by working backwards through visited states
    rev_states = visited_states[::-1]
    for j in range(len(rev_states)):
        state = rev_states[j]
        squares = state[0]
        if j == 0:
            theoretical_Q = game.p1_reward
        else:
            max_next_q = extract_max_possible_q(squares, model)
            theoretical_Q = ((1 - gamma) * theoretical_Q) + gamma * alpha * max_next_q
        training_data.append(
            [
                board_conversion_to_2x9(squares).cuda(),
                torch.tensor(theoretical_Q, dtype=torch.float).cuda(),
            ]
        )

    if i % 100 == 0 and i != 0:
        # train the training data for the model
        # predictive_qs = [model(squares) for squares, qs in training_data]
        # target_qs = [qs for squares, qs in training_data]

        train_data(training_data, model)

        win_count = 0
        tie_count = 0
        lose_count = 0

        recent_rewards = rewards[-100:]
        for reward in recent_rewards:
            if reward == 1:
                win_count += 1
            elif reward == 0.5:
                tie_count += 1
            else:
                lose_count += 1

        win_perc.append(win_count / 100)

        print("win % : ", format(win_count / 100, ".0%"))
        print("tie % : ", format(tie_count / 100, ".0%"))
        print("lose % : ", format(lose_count / 100, ".0%"))

        training_data = []

win_numbers = [int(item.strip("%")) for item in win_perc]

# data to be plotted
x = np.arange(0, len(win_numbers))

# plotting
plt.plot(x, win_numbers)
plt.show()
