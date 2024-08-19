import copy
from collections import deque
from cell import Cell
from player import Player
from wall import Wall
import heapq
from placeWall import PlaceWall


class Board:
    def __init__(self):
        self.board = []
        for i in range(17):
            self.board.append([])
            for j in range(17):
                self.board[i].append(Cell(i, j))
        self.winner = None
        self.game_over = False
        self.current_player = 1
        self.players = []

    def set_players(self, players):
        self.players = players

    def add_player(self, player):
        self.players.append(player)
        self.get_cell(player.get_x(), player.get_y()).set_player(player)

    def get_cell(self, x, y):
        return self.board[x][y]

    def copy(self):
        new_board = Board()
        new_board.board = [[cell.copy() for cell in row] for row in self.board]
        new_board.players = [player.copy() for player in self.players]
        return new_board

    def set_cell(self, x, y, cell):
        self.board[x][y] = cell

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_shortest_path(self, from_cell, to_cell):
        start = (from_cell.x, from_cell.y)
        goal = (to_cell.x, to_cell.y)
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Right, Down, Left, Up

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return len(path) - 1

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < 17 and 0 <= ny < 17:
                    if not self.wall_in_the_middle(current, (nx, ny)):  # Check if there's a wall in the middle
                        neighbor = self.get_cell(nx, ny)
                        if neighbor.is_empty() or neighbor.is_player():
                            new_cost = cost_so_far[current] + 1
                            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                                cost_so_far[(nx, ny)] = new_cost
                                priority = new_cost + self.heuristic(goal, (nx, ny))
                                heapq.heappush(frontier, (priority, (nx, ny)))
                                came_from[(nx, ny)] = current

        return []

    def wall_in_the_middle(self, from_coord, to_coord):
        x1, y1 = from_coord
        x2, y2 = to_coord
        if x1 == x2:  # Moving vertically
            mid_y = (y1 + y2) >> 1
            return self.get_cell(x1, mid_y).is_wall()
        elif y1 == y2:  # Moving horizontally
            mid_x = (x1 + x2) >> 1
            return self.get_cell(mid_x, y1).is_wall()
        return False

    def get_all_possible_moves(self, player):
        moves = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Right, Down, Left, Up

        for dx, dy in directions:
            nx, ny = player.get_x() + dx, player.get_y() + dy
            if 0 <= nx < 17 and 0 <= ny < 17:
                neighbor = self.get_cell(nx, ny)
                if neighbor.is_empty():
                    if not self.wall_in_the_middle((player.get_x(), player.get_y()), (nx, ny)):
                        moves.append(('move', nx, ny))
                # Handle jump over player
                elif neighbor.is_player():
                    jump_x, jump_y = player.get_x() + 2 * dx, player.get_y() + 2 * dy
                    if 0 <= jump_x < 17 and 0 <= jump_y < 17:
                        jump_cell = self.get_cell(jump_x, jump_y)
                        if jump_cell.is_empty():
                            if not self.wall_in_the_middle((nx, ny), (jump_x, jump_y)) and not self.wall_in_the_middle((player.get_x(), player.get_y()), (nx, ny)):
                                moves.append(('move', jump_x, jump_y))
                        else:
                            # Handle side jumps
                            side_directions = [(dx, 0), (0, dy)]
                            for sdx, sdy in side_directions:
                                side_x, side_y = nx + sdx, ny + sdy
                                if 0 <= side_x < 17 and 0 <= side_y < 17:
                                    side_cell = self.get_cell(side_x, side_y)
                                    if side_cell.is_empty() and not self.wall_in_the_middle((nx, ny), (side_x, side_y)):
                                        moves.append(('move', side_x, side_y))

        # Add possible wall placements within 3x3 squares surrounding the opponent and 1x1 around the player
        if player.get_num_of_walls() > 0:
            player_x, player_y = player.get_x(), player.get_y()
            opponent = self.get_opponent(player)
            opponent_x, opponent_y = opponent.get_x(), opponent.get_y()

            # 1x1 around the player
            for i in range(max(0, player_x - 1), min(17, player_x + 2)):
                for j in range(max(0, player_y - 1), min(17, player_y + 2)):
                    horizontal_wall = Wall('horizontal', (i, j), 3)
                    vertical_wall = Wall('vertical', (i, j), 3)
                    if self.can_place_wall(horizontal_wall):
                        moves.append(('wall', 'horizontal', i, j))
                    if self.can_place_wall(vertical_wall):
                        moves.append(('wall', 'vertical', i, j))

            # 3x3 around the opponent
            for i in range(max(0, opponent_x - 3), min(17, opponent_x + 3)):
                for j in range(max(0, opponent_y - 3), min(17, opponent_y + 3)):
                    horizontal_wall = Wall('horizontal', (i, j), 3)
                    vertical_wall = Wall('vertical', (i, j), 3)
                    if self.can_place_wall(horizontal_wall):
                        moves.append(('wall', 'horizontal', i, j))
                    if self.can_place_wall(vertical_wall):
                        moves.append(('wall', 'vertical', i, j))

        return moves

    def can_place_wall(self, wall):
        # Create a copy of the current board
        temp_board = self.copy()

        # Create a PlaceWall object to check if the wall is legal
        place_wall = PlaceWall(temp_board)

        # Check if the wall can be legally placed
        is_legal = place_wall.is_legal_wall(wall, self.players[0], self.players[1])

        return is_legal

    def set_wall_on_board(self, wall_locations):
        """
        Set the wall on the board by updating the appropriate cells.

        Args:
            wall_locations (list): List of coordinates (tuples) where the wall is placed.
        """
        for x, y in wall_locations:
            self.board[x][y].set_wall()

    def clean_wall_on_board(self, wall_locations):
        for x, y in wall_locations:
            self.board[x][y] = Cell(x, y)

    def execute_move(self, player, to_x, to_y):
        from_x, from_y = player.get_x(), player.get_y()
        self.get_cell(from_x, from_y).clean()
        self.get_cell(to_x, to_y).set_player(player)
        player.set_position((to_x, to_y))

    def execute_wall(self, wall):
        for (x, y) in wall.get_wall_location():
            self.set_cell(x, y, Cell(x, y, wall=True, empty=False))

    def get_opponent(self, player):
        if player.get_color() == 1:
            return self.players[1]
        elif player.get_color() == -1:
            return self.players[0]
        return None

    def display(self):
        for i in range(17):
            for j in range(17):
                cell = self.get_cell(i, j)
                if cell.is_player():
                    if cell.get_player() == 1:
                        print("P1", end=" ")
                    else:
                        print("P2", end=" ")
                elif cell.is_wall():
                    print("W", end=" ")
                else:
                    print(".", end=" ")
            print()

    def encode(self):
        res=''
        for i in range(17):
            for j in range(17):
                cell = self.get_cell(i, j)
                if cell.is_player():
                    if cell.get_player() == 1:
                        res+="P1"
                    else:
                        res+="P2"
                elif cell.is_wall():
                    res+="W"
                else:
                    res+="."
            res+="\n"
        return res            
            
    def is_game_over(self):
        for player in self.players:
            if (player.get_color() == 1 and player.get_x() == 16) or (player.get_color() == -1 and player.get_x() == 0):
                self.game_over = True
                self.winner = player
                return True
        return False

    def clone(self):
        new_board = Board()
        for i in range(17):
            for j in range(17):
                new_board.board[i][j] = Cell(self.board[i][j].x, self.board[i][j].y,
                                             self.board[i][j].player, self.board[i][j].wall,
                                             self.board[i][j].empty)
        new_board.players = [Player(player.position, player.color) for player in self.players]
        for player in new_board.players:
            new_board.get_cell(player.get_x(), player.get_y()).set_player(player)
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        return new_board

    def check_player_paths(self):
        for player in self.players:
            for i in range(17):
                to_cell = self.get_cell(16 if player.get_color() == 1 else 0, i)
                from_cell = self.get_cell(player.get_x(), player.get_y())
                if self.find_shortest_path(from_cell, to_cell):
                    break
            else:
                return False
        return True

    def has_free_path(self, from_cell, to_cell):
        queue = [(from_cell.x, from_cell.y)]
        visited = set()
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Right, Down, Left, Up

        while queue:
            current_x, current_y = queue.pop(0)
            if (current_x, current_y) == (to_cell.x, to_cell.y):
                return True
            if (current_x, current_y) in visited:
                continue
            visited.add((current_x, current_y))

            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < 17 and 0 <= ny < 17:
                    neighbor = self.get_cell(nx, ny)
                    if neighbor.is_empty() and (nx, ny) not in visited and not self.wall_in_the_middle(
                            (current_x, current_y), (nx, ny)):
                        queue.append((nx, ny))

        return False

    def get_winner(self):
        return self.winner

    def find_shortest_path_to_goal(self, player):
        from_cell = self.get_cell(player.get_x(), player.get_y())
        to_cells = [(16, i) for i in range(17)] if player.get_color() == 1 else [(0, i) for i in range(17)]
        shortest_len = float('inf')
        if player.get_color() == 1 and (player.get_x() == 16 or from_cell.get_x() == 16):
            return 0
        if player.get_color() == -1 and (player.get_x() == 0 or from_cell.get_x() == 0):
            return 0

        for to_cell in to_cells:
            to_cell_obj = self.get_cell(to_cell[0], to_cell[1])
            path = self.find_shortest_path(from_cell, to_cell_obj)
            if path:
                shortest_len = min(shortest_len, path)
            if shortest_len == 1:
                return 1

        return shortest_len

    def get_player(self, color):
        for player in self.players:
            if player.get_color() == color:
                return player
        return None

    
    def get_state(self):
        """
        Returns a simplified representation of the current state of the board.
        """
        print(self)
        # exit(0)
        # state = {
        #     'white_player_pos': self.white_player.get_position(),
        #     'black_player_pos': self.black_player.get_position(),
        #     'walls': self.get_walls(),
        #     'turn': self.current_player_turn
        # }
        # return state
        state = {
            'white_player_pos': self.white_player.get_position(),
            'black_player_pos': self.black_player.get_position(),
            'walls': self.get_walls(),
            'turn': self.current_player_turn
        }
        return state

    # def get_walls(self):
    #     walls = []
    #     for i in range(17):
    #         for j in range(17):
    #             cell = self.get_cell(i, j)
    #             if cell.is_wall():
    #                 walls.append((i, j, cell.get_wall().orientation))
    #     return tuple(walls)
