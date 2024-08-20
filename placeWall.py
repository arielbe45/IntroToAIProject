from cell import Cell


class PlaceWall:
    def __init__(self, board):
        self.board = board

    def place_wall(self, wall, Wplayer, Bplayer):
        """
        Places a wall on the board if the placement is legal.

        :param wall: The wall object to place.
        :param Wplayer: The white player object.
        :param Bplayer: The black player object.
        :return: True if the wall was placed successfully, else False.
        """
        arr = wall.get_wall_location()

        if not self.in_board(arr):
            return False

        if not self.possible_placement_in_board(wall):
            return False

        if self.already_caught(arr):
            return False

        if not self.legal_wall(wall, Wplayer, Bplayer):
            return False

        self.set_wall_on_board(arr)
        return True

    def is_legal_wall(self, wall, Wplayer, Bplayer):
        """
        Places a wall on the board if the placement is legal.

        :param wall: The wall object to place.
        :param Wplayer: The white player object.
        :param Bplayer: The black player object.
        :return: True if the wall was placed successfully, else False.
        """
        arr = wall.get_wall_location()

        if not self.in_board(arr):
            return False

        if not self.possible_placement_in_board(wall):
            return False

        if self.already_caught(arr):
            return False

        if not self.legal_wall(wall, Wplayer, Bplayer):
            return False
        return True

    def in_board(self, arr):
        """
        Checks if the wall positions are within the board boundaries.

        :param arr: The wall positions.
        :return: True if all positions are within the board, else False.
        """
        for (x, y) in arr:
            if x > 16 or x < 0 or y > 16 or y < 0:
                return False
        return True

    def possible_placement_in_board(self, wall):
        """
        Checks if the wall placement is on the valid lines.

        :param wall: The wall object.
        :return: True if the wall placement is valid, else False.
        """
        if wall.get_wall_block_design() == '|':
            if wall.get_start_position()[1] % 2 == 0 or wall.get_start_position()[0] % 2 == 1:
                return False

        if wall.get_wall_block_design() == '-':
            if wall.get_start_position()[1] % 2 == 1 or wall.get_start_position()[0] % 2 == 0:
                return False
        return True

    def already_caught(self, arr):
        """
        Checks if the wall positions are already occupied.

        :param arr: The wall positions.
        :return: True if any position is occupied, else False.
        """
        for (x, y) in arr:
            if not self.board.get_cell(x, y).is_empty():
                return True
        return False

    def legal_wall(self, wall, Wplayer, Bplayer):
        """
        Checks if placing the wall would still allow both players a path to their goals.

        :param wall: The wall object.
        :param Wplayer: The white player object.
        :param Bplayer: The black player object.
        :return: True if both players still have a path to their goals, else False.
        """

        arr = wall.get_wall_location()
        # Temporarily place the wall
        self.set_wall_on_board(arr)

        # Check paths for both players
        flag = self.check_player_paths(Wplayer, Bplayer)

        if not flag:
            # Remove the temporary wall if placement is illegal
            self.clean_wall_on_board(arr)
        return flag

    def set_wall_on_board(self, arr):
        """
        Sets a wall on the board at the given positions.

        :param arr: The wall positions.
        """
        for (x, y) in arr:
            self.board.set_cell(x, y, Cell(x, y, wall=True, empty=False))

    def clean_wall_on_board(self, arr):
        """
        Cleans a wall from the board at the given positions.

        :param arr: The wall positions.
        """
        for (x, y) in arr:
            self.board.set_cell(x, y, Cell(x, y))
            self.board.get_cell(x, y).clean()

    def check_player_paths(self, Wplayer, Bplayer):
        """
        Checks if both players have a valid path to their goals.

        :param Wplayer: The white player object.
        :param Bplayer: The black player object.
        :return: True if both players have a path, else False.
        """
        for i in range(0, 17, 2):
            from_cell = self.board.get_cell(Wplayer.get_position()[0], Wplayer.get_position()[1])
            to_cell = Cell(16, i)
            if self.board.has_free_path(from_cell, to_cell):
                break
        else:
            return False

        for i in range(0, 17, 2):
            from_cell = self.board.get_cell(Bplayer.get_position()[0], Bplayer.get_position()[1])
            to_cell = Cell(0, i)
            if self.board.has_free_path(from_cell, to_cell):
                break
        else:
            return False

        return True
