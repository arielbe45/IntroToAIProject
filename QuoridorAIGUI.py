import tkinter as tk
from tkinter import messagebox
from game import Game
from move import Move
from placeWall import PlaceWall
from wall import Wall
from minimax import Minimax

class QuoridorAIGUI:

    counter = 0

    def __init__(self, root):
        self.root = root
        self.root.title("Quoridor AI vs AI Game")
        self.game = Game()
        self.selected_position = None
        self.wall_mode = False
        self.ai_running = False

        # Create the canvas for the game board
        self.canvas = tk.Canvas(root, width=680, height=680)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Create buttons for player actions
        self.start_button = tk.Button(root, text="Start AI vs AI", command=self.start_ai_game)
        self.start_button.grid(row=1, column=1)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.grid(row=1, column=2)

        # Draw the initial board
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(17):
            for j in range(17):
                cell = self.game.board.get_cell(j, i)  # Swap i and j for correct orientation
                x0 = i * 40
                y0 = j * 40
                x1 = x0 + 40
                y1 = y0 + 40
                if cell.is_player():
                    if cell.get_player() == 1:
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill="green")
                    else:
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill="black")
                elif cell.is_wall():
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="red")
                elif i % 2 == 1 or j % 2 == 1:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="gray")
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1)

    def start_ai_game(self):
        if not self.ai_running:
            self.ai_running = True
            self.run_ai_turn()

    def run_ai_turn(self):
        if not self.ai_running:
            return

        if self.game.is_game_over() == 0:
            QuoridorAIGUI.counter = QuoridorAIGUI.counter + 1
            current_player = self.game.players[self.game.current_player_turn]
            if current_player.get_color() == 1:
                self.game.minimax = Minimax(self.game.board, 3)  # Adjust the depth as needed
            else:
                self.game.minimax = Minimax(self.game.board, 3)  # Adjust the depth as needed

            best_move = self.game.minimax.run(current_player)
            if QuoridorAIGUI.counter == 4:
                wall = Wall("horizontal", (5, 8), 3)
                self.game.execute_wall(wall)
            elif best_move[0] == 'move':
                self.game.execute_move(Move(current_player, self.game.board,
                                            self.game.board.get_cell(current_player.get_position()[0],
                                                                     current_player.get_position()[1]),
                                            self.game.board.get_cell(best_move[1], best_move[2])))
            elif best_move[0] == 'wall':
                wall = Wall(best_move[1], (best_move[2], best_move[3]), 3)
                if not PlaceWall(self.game.board).place_wall(wall, self.game.white_player, self.game.black_player):
                    print("AI made an invalid wall placement, something went wrong.")
                else:
                    self.game.execute_wall(wall)

            self.game.switch_player()
            self.draw_board()
            self.root.after(1000, self.run_ai_turn)  # Delay to visualize the moves
        else:
            self.ai_running = False
            winner = self.game.is_game_over()
            messagebox.showinfo("Game Over", f"Player {winner} wins!")
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuoridorAIGUI(root)
    root.mainloop()
