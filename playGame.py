import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
from tictactoe import TicTacToe

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        # try:
        #     self.model = load_model('tictactoe_model.keras')
        # except Exception as e:
        #     print(f"Failed to load model: {e}")
        #     self.model = None
        self.starting_player = "ai"
        self.opponent = 'minimax'
        self.env = TicTacToe()
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack()
        for row in range(3):
            for col in range(3):
                button = tk.Button(frame, text=" ", font='normal 20 bold', height=2, width=5,
                                   command=lambda row=row, col=col: self.on_button_click(row, col))
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

    def reset_game(self):
        self.env.reset()
        for row in range(3):
            for col in range(3):
                self.buttons[row][col].config(text=" ", state=tk.NORMAL)
        if self.starting_player == "ai":
            if self.opponent == 'model':
                self.ai_move()
            elif self.opponent == 'minimax':
                row, col = self.env.best_move()
                self.env.make_move(row, col)
                self.buttons[row][col].config(text="O")
                
    def set_opponent(self, opponent_type):
        """Set the opponent type to either 'model' or 'minimax'."""
        self.opponent = opponent_type
        
    def set_starting_player(self, player_type):
        """Set the starting player to either 'human' or 'ai'."""
        self.starting_player = player_type

    def on_button_click(self, row, col):
        if self.env.board[row, col] != 0:
            return
        self.env.make_move(row, col)
        self.buttons[row][col].config(text="X")
        if not self.check_game_over():
            if self.opponent == 'model':
                self.ai_move()
            elif self.opponent == 'minimax':
                row, col = self.env.best_move()
                self.env.make_move(row, col)
                self.buttons[row][col].config(text="O")
                self.check_game_over()

    def ai_move(self):
        state = self.env.get_state()
        row, col = self.choose_action(state, self.model)
        self.env.make_move(row, col)
        self.buttons[row][col].config(text="O")
        self.check_game_over()

    def choose_action(self, state, model):
        q_values = model.predict(state.reshape(1, -1))
        flat_q_values = q_values.flatten()
        available_actions = np.argsort(flat_q_values)[::-1]
        for action in available_actions:
            row, col = divmod(action, 3)
            if state[row*3 + col] == 0:
                return (row, col)

    def check_game_over(self):
        if self.env.check_winner(1):
            messagebox.showinfo("Game Over", "You win!")
            self.reset_game()
            return True
        elif self.env.check_winner(-1):
            messagebox.showinfo("Game Over", "AI wins!")
            self.reset_game()
            return True
        elif self.env.is_draw():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
            return True
        return False

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
