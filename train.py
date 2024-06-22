import numpy as np
import random
from collections import deque
from tictactoe import TicTacToe
from model import create_model
from keras.models import load_model

# Replay buffer parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Hyperparameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EVAL_FREQ = 100

def choose_action(state, model, epsilon):
    if np.random.rand() < epsilon:
        return random.choice([(r, c) for r in range(3) for c in range(3) if state[r*3 + c] == 0])
    q_values = model.predict(state.reshape(1, -1))
    flat_q_values = q_values.flatten()
    available_actions = np.argsort(flat_q_values)[::-1]
    for action in available_actions:
        row, col = divmod(action, 3)
        if state[row*3 + col] == 0:
            return (row, col)

def play_game(model):
    env = TicTacToe()
    env.reset()
    state = env.get_state()
    done = False
    while not done:
        action = choose_action(state, model, epsilon=0)
        row, col = action
        reward = env.make_move(row, col)
        env.render()
        next_state = env.get_state()
        done = reward != 0 or env.is_draw()
        state = next_state
    if reward == 1:
        print("player 1 wins!")
    elif reward == -1:
        print("player -1 wins!")
    else:
        print("It's a draw!")

def minimax_move(env):
    return env.best_move()

def train(model, episodes, gamma, epsilon, epsilon_min, epsilon_decay, eval_freq=100):
    env = TicTacToe()
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epsilon_history = []

    for e in range(episodes):
        env.reset()
        training_against_minimax = random.choice([True, False])  # Hybrid approach
        if training_against_minimax:
            print("Training against minimax")
        else:
            print("Self-play")
        
        neural_net_starts = random.choice([True, False])

        if not neural_net_starts:
            print("Opponent starts")
            env.current_player = -1
        else:
            print("Neural network starts")
            env.current_player = 1
        state = env.get_state()
        done = False
        while not done:
            if env.current_player == 1:  # Neural network's turn
                action = choose_action(state, model, epsilon)
            else:  # Opponent's turn
                if training_against_minimax:
                    action = minimax_move(env)
                else:
                    action = choose_action(state, model, epsilon)  # Self-play

            if action:
                row, col = action
                reward = env.make_move(row, col)
                next_state = env.get_state()
                done = reward != 0 or env.is_draw()
                reward = 1 if reward == 1 else 0 if reward == 0 else -1
                if env.current_player == -1:  # Only add to buffer when NN plays
                    replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            for state, action, reward, next_state, done in batch:
                target = reward
                if not done:
                    target = reward + gamma * np.amax(model.predict(next_state.reshape(1, -1)))
                target_f = model.predict(state.reshape(1, -1))
                row, col = action
                target_f[0][row*3 + col] = target
                model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (e+1) % eval_freq == 0:
            print(f"Episode {e+1}/{episodes} - Epsilon: {epsilon}")
            play_game(model)

        epsilon_history.append(epsilon)
        print(f"Episode {e+1}/{episodes} completed")

    model.save("tictactoe_model.keras")


if __name__ == "__main__":
    try:
        model = load_model('tictactoe_model.keras')
    except:
        model = create_model()
    train(model, episodes=10000, gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, eval_freq=EVAL_FREQ)
