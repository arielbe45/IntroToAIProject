class Cell:
    def __init__(self, x, y, player=None, wall=False, empty=True):
        self.wall = wall
        self.player = player
        self.empty = empty
        self.x = x
        self.y = y

    def is_wall(self):
        return self.wall

    def is_player(self):
        return self.player is not None

    def set_empty(self, flag):
        self.empty = flag

    def copy(self):
        new_cell = Cell(self.x, self.y, self.player, self.wall, self.empty)
        return new_cell

    def get_player(self):
        return self.player.get_color()

    def get_wall(self):
        return self.wall
    
    def is_empty(self):
        return self.player is None and not self.wall

    def set_wall(self):
        self.wall = True
        self.empty = False
        self.player = None

    def set_player(self, player):
        self.player = player
        self.empty = False
        self.wall = False

    def clean(self):
        self.empty = True
        self.wall = False
        self.player = None

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_position(self):
        return (self.x, self.y)
