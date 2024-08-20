class Player:
    def __init__(self, start_position, color, num_of_walls=10):
        self.position = start_position
        self.x = self.position[0]
        self.y = self.position[1]
        self.color = color
        self.walls = num_of_walls  # Assuming each player starts with 10 walls

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = (self.x, self.y)

    def get_position(self):
        return self.position

    def get_x(self):
        return self.x

    def copy(self):
        new_player = Player(self.position, self.color, self.walls)
        return new_player

    def get_y(self):
        return self.y

    def set_position(self, new_position):
        self.position = new_position
        self.x = self.position[0]
        self.y = self.position[1]

    def set_x(self, new_x):
        self.x = new_x
        self.position = (self.x, self.y)

    def set_y(self, new_y):
        self.y = new_y
        self.position = (self.x, self.y)

    def get_color(self):
        return self.color

    def get_num_of_walls(self):
        return self.walls

    def use_wall(self):
        if self.walls > 0:
            self.walls -= 1
