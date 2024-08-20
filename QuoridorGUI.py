import tkinter as tk
from tkinter import simpledialog, messagebox
from board import Board
from game import Game
from player import Player
from wall import Wall
from move import Move
from cell import Cell
from placeWall import PlaceWall


class QuoridorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quoridor Game")
        self.game = Game()
        self.selected_position = None
        self.wall_mode = False

        # Create the canvas for the game board
        self.canvas = tk.Canvas(root, width=680, height=680)
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Create buttons for player actions
        self.move_button = tk.Button(root, text="Move", command=self.activate_move_mode)
        self.move_button.grid(row=1, column=0)

        self.wall_button = tk.Button(root, text="Place Wall", command=self.activate_wall_mode)
        self.wall_button.grid(row=1, column=1)

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
                elif i == 1 or i == 3 or i == 5 or i == 7 or i == 9 or i == 11 or i == 13 or i == 15:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="gray")
                elif j == 1 or j == 3 or j == 5 or j == 7 or j == 9 or j == 11 or j == 13 or j == 15:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="gray")
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1)

    def on_canvas_click(self, event):
        x = event.y // 40  # Swap x and y for correct orientation
        y = event.x // 40  # Swap x and y for correct orientation
        if self.wall_mode:
            self.place_wall(x, y)
        else:
            self.move_player(x, y)

    def activate_move_mode(self):
        self.wall_mode = False

    def activate_wall_mode(self):
        self.wall_mode = True

    def move_player(self, x, y):
        current_player = self.game.players[self.game.current_player_turn]
        from_square = self.game.board.get_cell(current_player.get_position()[0], current_player.get_position()[1])
        to_square = self.game.board.get_cell(x, y)

        move = Move(current_player, self.game.board, from_square, to_square)
        if move.is_valid_move():
            self.game.execute_move(move)
            self.game.switch_player()
            self.draw_board()
            self.check_game_over()
        else:
            messagebox.showerror("Invalid Move", f"The move to ({x}, {y}) is not valid. Try again.")

    def place_wall(self, x, y):
        current_player = self.game.players[self.game.current_player_turn]
        if current_player.get_num_of_walls() == 0:
            messagebox.showerror("No Walls Left", "You have no walls left to place.")
            return

        orientation = simpledialog.askstring("Wall", "Enter wall orientation (h for horizontal, v for vertical):")
        if orientation == 'h':
            orientation = "horizontal"
        elif orientation == 'v':
            orientation = "vertical"
        else:
            messagebox.showerror("Invalid Input", "Orientation must be 'h' or 'v'.")
            return

        wall = Wall(orientation, (x, y), 3)
        place_wall = PlaceWall(self.game.board)

        if place_wall.place_wall(wall, self.game.white_player, self.game.black_player):
            self.game.execute_wall(wall)
            self.game.switch_player()
            self.draw_board()
            self.check_game_over()
        else:
            messagebox.showerror("Invalid Wall Placement", f"The wall placement at ({x}, {y}) is not valid. Try again.")

    def check_game_over(self):
        winner = self.game.is_game_over()
        if winner != 0:
            messagebox.showinfo("Game Over", f"Player {winner} wins!")
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = QuoridorGUI(root)
    root.mainloop()
