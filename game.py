import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_winner(self.current_player):
                return 1.0  # reward for winning
            self.current_player = -1 if self.current_player == 1 else 1  # switch player
            return 0.0  # neutral reward for valid move
        elif self.check_draw():
            return -0.5  # small negative reward for draw
        else:
            return -1.0  # negative reward for invalid move

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def get_state(self):
        return self.board.flatten()

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def render(self):
        print(self.board)
        print()

    def minimax(self, depth, is_maximizing):
        if self.check_winner(1):
            return 10 - depth  # Adjust score based on depth for player 1 win
        if self.check_winner(-1):
            return depth - 10  # Adjust score based on depth for player -1 win
        if self.is_draw():
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for (r, c) in self.available_moves():
                self.board[r, c] = 1  # Assuming 1 is the maximizing player
                score = self.minimax(depth + 1, False)
                self.board[r, c] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for (r, c) in self.available_moves():
                self.board[r, c] = -1  # Assuming -1 is the minimizing player
                score = self.minimax(depth + 1, True)
                self.board[r, c] = 0
                best_score = min(score, best_score)
            return best_score

    def best_move(self):
        best_score = -float('inf')
        move = None
        for (r, c) in self.available_moves():
            self.board[r, c] = 1
            score = self.minimax(0, False)
            self.board[r, c] = 0
            if score > best_score:
                best_score = score
                move = (r, c)
        return move

        def ai_move(self):
            move = self.best_move()
            if move:
                self.make_move(move[0], move[1])
