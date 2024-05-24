import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tictactoe import Board, Game
from random_agent import RandomAgent

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


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


class TicTacToeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 18), nn.ReLU(), nn.Linear(18, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # x = torch.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        # x = self.fc2(x)              # Apply the second linear transformation
        return logits


model = TicTacToeNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action


def nn_play(game, model):
    possible_states = list()
    # Make hashes of each possible next game state
    for x in range(3):
        for y in range(3):
            # print(f"x: {x} | y: {y} | value: {game.board.squares[x][y]}")
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


# random_agent = RandomAgent()
# visited_states = []
# game = Game()
# while not game.done:
#     x, y, Q = nn_play(game, model)
#     game.play(x, y)
#     visited_state = game.board.squares
#     visited_states.append((visited_state, Q))
#     if game.done:
#         break
#     random_agent.play(game)

# print(visited_states)
# print(game.board.show())

# def train(data, model, loss_fn, optimizer):
#     model.train()
#     for i in something:
#         X = board_conversion_to_2x9(board.squares)

#         # Compute prediction error
#         move_qvalues = model(X)
#         move_we_pick = epsilon_greedy_policy(move_qvalues, .9)
#         # Somehow keep track of move we made / state we picked
#         # repeat all this for a whole game
#         # calculate Q values for all the states we visited based on the Q value formula
#         loss = loss_fn(move_qvalues, Q_vals_we_calc_from_formula)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

criterion = torch.nn.MSELoss()


def train_data(data, model):
    num_epochs = 5

    for epoch in range(num_epochs):
        total_loss = 0
        for squares, q in data:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(squares)

            # Compute the loss
            loss = criterion(output, q)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            total_loss += loss.item()


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

for i in range(10001):
    # print(f"Game {i}")
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
        # print(training_data)
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

        print("win % : ", format(win_count / 100, ".0%"))
        print("tie % : ", format(tie_count / 100, ".0%"))
        print("lose % : ", format(lose_count / 100, ".0%"))

        training_data = []
