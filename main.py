import numpy as np
import math
from treelib import Node, Tree

# todo:
# fix up Board nodes in tree
# implement priority queue


class Board:

    # the total number of squares in the world
    #   tiles = SIZE - 1
    #   sqrt(SIZE) = width = height
    SIZE = 0    # the total number of squares in the world

    # the location of the blank square
    # blank = 0

    # the array of tiles
    tiles = []

    def __init__(self, size):
        self.SIZE = size
        self.WIDTH = int(math.sqrt(size))

        # generate n squares at random locations
        self.tiles = np.random.choice(size, size, replace=False)

        # the square set to zero will be the blank square
        # self.blank = np.where(self.tiles == 0)

    # get location of the empty square
    def empty(self):
        return int(np.where(self.tiles == 0)[0])

    # swap two tiles
    def swap(self, a, b):
        # check for invalid indexes
        if a < 0 or a > self.SIZE - 1 \
                or b < 0 or b > self.SIZE - 1:
            return False

        temp = self.tiles[a]
        self.tiles[a] = self.tiles[b]
        self.tiles[b] = temp
        return True

    # move the blank up
    def up(self):
        e = self.empty()

        # make sure there is a row above
        if e < self.WIDTH:
            return False

        self.swap(e, e - self.WIDTH)
        return True

    # move the blank down
    def down(self):
        e = self.empty()

        # make sure there is a row below
        if e > self.SIZE - 1 - self.WIDTH:
            return False

        self.swap(e, e + self.WIDTH)
        return True

    # move the blank left
    def left(self):
        e = self.empty()

        # make sure there is a column to the left
        if e % self.WIDTH == 0:
            return False

        self.swap(e, e - 1)
        return True

    # move the blank right
    def right(self):
        e = self.empty()

        # make sure there is a column to the right
        if (e + 1) % self.WIDTH == 0:
            return False

        self.swap(e, e + 1)
        return True

    # heuristic 1 - number of misplaced tiles
    def h1(self):
        cost = 0
        # loop through tiles, check if misplaced
        for i in range(0, self.SIZE):
            if self.tiles[i] != i and self.tiles[i] != 0:
                cost += 1
        return cost

    # heuristic 2 - total Manhattan distance (i.e. 1-norm)
    def h2(self):
        cost = 0
        # loop through tiles, calculate distance from correct square
        for i in range(0, self.SIZE):
            if self.tiles[i] != 0:
                # how many rows off left/right
                dist_lr = abs((self.tiles[i] % self.WIDTH) - (i % self.WIDTH))
                # how many columns off up/down
                dist_ud = abs(math.floor(self.tiles[i] / self.WIDTH) - math.floor(i / self.WIDTH))
                cost += dist_lr + dist_ud
        return cost

    def info(self):
        print("Size: ", self.SIZE, " Width: ", self.WIDTH, " Empty tile: ", self.empty())
        print(self.tiles.reshape((self.WIDTH, self.WIDTH)))
        print("H1 cost: ", self.h1(), " H2 cost: ", self.h2())


if __name__ == '__main__':

    world = Board(3*3)

    world.info()

    world.up()
    world.up()
    world.up()

    world.info()

    world.left()
    world.left()
    world.left()

    world.info()

    world.right()
    world.right()
    world.right()

    world.info()

    world.down()
    world.down()
    world.down()

    world.info()

