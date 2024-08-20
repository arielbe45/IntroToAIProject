class Move:
    def __init__(self, player, board, from_square, to_square):
        self.player = player
        self.board = board
        self.from_square = from_square
        self.to_square = to_square

    def is_valid_move(self):
        if not self.is_valid_cells():
            return False
        # move one square - no wall no player
        if self.valid_simple_move():
            return True
        # wall in the way
        if self.wall_in_the_middle():
            return False
        # jump over player
        if self.valid_jump():
            return True
        # diagonal move
        if self.valid_diagonal_move():
            return True
        return False

    def is_valid_cells(self):
        # check if the start and end cells are legal
        if self.to_square.get_x() < 0 or self.to_square.get_y() < 0:
            return False
        if self.to_square.get_x() > 16 or self.to_square.get_y() > 16:
            return False
        if self.from_square.is_empty():
            return False
        if not (self.from_square.is_player() and self.from_square.get_player() == self.player.get_color()):
            return False
        if self.to_square.get_x() == self.from_square.get_x() and self.to_square.get_y() == self.from_square.get_y():
            return False

        return True

    def valid_diagonal_move(self):
        other_player_location = None
        token = None
        dx = self.from_square.get_x() - self.to_square.get_x()
        dy = self.from_square.get_y() - self.to_square.get_y()
        if abs(dx) != 2 or abs(dy) != 2:
            return False
        # to_cell is empty
        if not self.to_square.is_empty():
            return False
        # locate the other player
        if not self.board.get_cell(self.from_square.get_x() - dx, self.from_square.get_y()).is_empty():
            other_player_location = (self.from_square.get_x() - dx, self.from_square.get_y())
            token = 1
        elif not self.board.get_cell(self.from_square.get_x(), self.from_square.get_y() - dy).is_empty():
            other_player_location = (self.from_square.get_x(), self.from_square.get_y() - dy)
            token = 2
        elif not self.board.get_cell(self.from_square.get_x(), self.from_square.get_y() + dy).is_empty():
            other_player_location = (self.from_square.get_x(), self.from_square.get_y() + dy)
            token = 3
        elif not self.board.get_cell(self.from_square.get_x() + dx, self.from_square.get_y()).is_empty():
            other_player_location = (self.from_square.get_x() + dx, self.from_square.get_y())
            token = 4
        if other_player_location is None:
            return False
        # check if there is a wall in the middle between the players
        if self.wall_in_the_middle_two_player(token):
            return False
        # check if there is a wall between the other player and the to_cell
        if self.wall_in_the_middle_diagonal(token, dx, dy):
            return False

        if self.to_square.get_x() != other_player_location[0] and self.to_square.get_y() != other_player_location[1]:
            return False
        return True

    def wall_in_the_middle_diagonal(self, token, dx, dy):
        if token == 1 and dy == 2:
            if self.board.get_cell(self.from_square.get_x() - 2, self.from_square.get_y() + 1).is_wall():
                return True

        if token == 1 and dy == -2:
            if self.board.get_cell(self.from_square.get_x() - 2, self.from_square.get_y() - 1).is_wall():
                return True

        if token == 2 and dx == 2:
            if self.board.get_cell(self.from_square.get_x() + 1, self.from_square.get_y() - 2).is_wall():
                return True

        if token == 2 and dx == -2:
            if self.board.get_cell(self.from_square.get_x() - 1, self.from_square.get_y() - 2).is_wall():
                return True

        if token == 3 and dx == 2:
            if self.board.get_cell(self.from_square.get_x() + 1, self.from_square.get_y() + 2).is_wall():
                return True

        if token == 3 and dx == -2:
            if self.board.get_cell(self.from_square.get_x() - 1, self.from_square.get_y() + 2).is_wall():
                return True

        if token == 4 and dy == 2:
            if self.board.get_cell(self.from_square.get_x() + 2, self.from_square.get_y() + 1).is_wall():
                return True

        if token == 4 and dy == -2:
            if self.board.get_cell(self.from_square.get_x() + 2, self.from_square.get_y() - 1).is_wall():
                return True

        return False

    def wall_in_the_middle_two_player(self, token):
        if token == 1:
            if self.board.get_cell(self.from_square.get_x() - 1, self.from_square.get_y()).is_wall():
                return True
        if token == 2:
            if self.board.get_cell(self.from_square.get_x(), self.from_square.get_y() - 1).is_wall():
                return True
        if token == 3:
            if self.board.get_cell(self.from_square.get_x(), self.from_square.get_y() + 1).is_wall():
                return True
        if token == 4:
            if self.board.get_cell(self.from_square.get_x() + 1, self.from_square.get_y()).is_wall():
                return True

    def wall_in_the_middle(self):
        # check if there is a wall in the middle of the move
        dx = self.from_square.get_x() - self.to_square.get_x()
        dy = self.from_square.get_y() - self.to_square.get_y()
        if dx == 0 and abs(dy) == 2:
            if self.board.get_cell(self.from_square.get_x(), self.from_square.get_y() - int(dy / 2)).is_wall():
                return True
        if self.from_square.get_y() == self.to_square.get_y() and abs(
                self.from_square.get_x() - self.to_square.get_x()) == 2:
            if self.board.get_cell((self.from_square.get_x() - int(dx / 2)), self.from_square.get_y()).is_wall():
                return True
        return False

    def valid_simple_move(self):
        dx = self.from_square.get_x() - self.to_square.get_x()
        dy = self.from_square.get_y() - self.to_square.get_y()
        # check if the move is legal
        if self.from_square.get_x() == self.to_square.get_x() and abs(
                self.from_square.get_y() - self.to_square.get_y()) == 2:
            if self.board.get_cell(self.from_square.get_x(), (self.from_square.get_y() - int(dy / 2))).is_empty():
                # check if end cell is empty
                if self.to_square.is_empty():
                    return True
        if self.from_square.get_y() == self.to_square.get_y() and abs(
                self.from_square.get_x() - self.to_square.get_x()) == 2:
            if self.board.get_cell(int(self.from_square.get_x() - int(dx / 2)),
                                   int(self.from_square.get_y())).is_empty():
                # check if end cell is empty
                if self.to_square.is_empty():
                    return True

        return False

    def valid_jump(self):
        dx = self.from_square.get_x() - self.to_square.get_x()
        dy = self.from_square.get_y() - self.to_square.get_y()
        # check if the move is legal
        if self.from_square.get_x() == self.to_square.get_x() and abs(
                self.from_square.get_y() - self.to_square.get_y()) == 4:
            if self.board.get_cell(self.from_square.get_x(), (self.from_square.get_y() - int(dy / 4))).is_empty():
                if self.board.get_cell(
                        (self.from_square.get_x(), (self.from_square.get_y() - int(dy / 2)))).is_player() is not None:
                    if self.board.get_cell(
                            self.from_square.get_x(), (self.from_square.get_y() - int(3 * (dy / 4)))).is_empty():
                        # check if end cell is empty
                        if self.to_square.is_empty():
                            return True
        if self.from_square.get_y() == self.to_square.get_y() and abs(
                self.from_square.get_x() - self.to_square.get_x()) == 4:
            if self.board.get_cell((self.from_square.get_x() - int(dx / 4)), self.from_square.get_y()).is_empty():
                if self.board.get_cell(
                        (self.from_square.get_x() - int(dx / 2)), self.from_square.get_y()).is_player() is not None:
                    if self.board.get_cell(
                            (self.from_square.get_x() - int(3 * (dx / 4))), self.from_square.get_y()).is_empty():
                        # check if end cell is empty
                        if self.to_square.is_empty():
                            return True
        return False
