class Wall:
    def __init__(self, orientation, start_position, length):
        self.orientation = orientation
        self.start_position = start_position
        self.length = length
        self.end_position = self.calculate_end_position()

    def calculate_end_position(self):
        if self.orientation == 'horizontal':
            return self.start_position[0], self.start_position[1] + self.length
        elif self.orientation == 'vertical':
            return self.start_position[0] + self.length, self.start_position[1]
        else:
            raise ValueError("Invalid orientation. Must be 'horizontal' or 'vertical'.")

    def get_orientation(self):
        return self.orientation

    def get_start_position(self):
        return self.start_position

    def get_end_position(self):
        return self.end_position

    def get_length(self):
        return self.length

    def get_wall_location(self):
        #return an arry of tuples with the coordinates of the wall
        if self.orientation == 'horizontal':
            return [(self.start_position[0], self.start_position[1]+i) for i in range(self.length)]
        elif self.orientation == 'vertical':
            return [(self.start_position[0]+i, self.start_position[1]) for i in range(self.length)]
        else:
            raise ValueError("Invalid orientation. Must be 'horizontal' or 'vertical'.")

    def get_wall_block_design(self):
        #return the wall block
        if self.orientation == 'horizontal':
            return '-'
        elif self.orientation == 'vertical':
            return '|'
        else:
            raise ValueError("Invalid orientation. Must be 'horizontal' or 'vertical'.")

    def get_what_wall_block(self):
        #if the wall is horizontal the wall block location above and below the wall
        #if the wall is vertical the wall block location to the left and right of the wall
        if self.orientation == 'horizontal':
            block= [(self.start_position[0] + i, self.start_position[1] + 1) for i in range(self.length)]
            block+= [(self.start_position[0] + i, self.start_position[1] - 1) for i in range(self.length)]
        elif self.orientation == 'vertical':
            block = [(self.start_position[0] + 1, self.start_position[1] + i) for i in range(self.length)]
            block+= [(self.start_position[0] - 1, self.start_position[1] + i) for i in range(self.length)]
        else:
            raise ValueError("Invalid orientation. Must be 'horizontal' or 'vertical'.")