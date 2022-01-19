import numpy as np
import math
from treelib import Node, Tree
import copy

# todo:
# fix up Board nodes in tree
# implement priority queue


class Board:

    def __init__(self, size, conf="rand"):
        self.SIZE = size + 1  # size = total number of spaces on the board
        self.WIDTH = int(math.sqrt(self.SIZE))

        if conf == "rand":
            # generate n squares at random locations
            self.tiles = np.random.choice(self.SIZE, self.SIZE, replace=False)
        else:
            # I am using 0 to represent the empty square
            conf = conf.replace("b", "0")
            # split the string and build the array
            self.tiles = np.array(list(map(int, conf.split())))

    # HELPERS

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

    # MOVES

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

    # HEURISTICS

    # heuristic 1 - number of misplaced tiles
    def h1(self):
        cost = 0
        # loop through tiles, check if misplaced
        for i in range(0, self.SIZE):
            if self.tiles[i] != i and self.tiles[i] != 0:
                cost += 1
        return cost

    # heuristic 2 - total Euclidian distance (L2 norm)
    # dominates h1
    def h2(self):
        cost = 0
        # loop through tiles, calculate distance from correct square
        for i in range(0, self.SIZE):
            if self.tiles[i] != 0:
                # how many rows off left/right
                dist_lr = (self.tiles[i] % self.WIDTH) - (i % self.WIDTH)
                # how many columns off up/down
                dist_ud = math.floor(self.tiles[i] / self.WIDTH) - math.floor(i / self.WIDTH)
                cost += math.sqrt(dist_lr*dist_lr + dist_ud*dist_ud)
        return cost

    # heuristic 3 - total Manhattan distance (i.e. 1-norm)
    # dominates h2
    def h3(self):
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

    # DISPLAY

    # return unique identifier for this board configuration
    def id(self):
        # return int(np.core.defchararray.join('', self.tiles.tolist()))
        # generate an array with strings
        ID = ""
        for i in self.tiles:
            ID += str(i)

        return int(ID)





    # print some basic info about the board
    def info(self):
        print("Board ID: ", self.id())
        print("Size: ", self.SIZE, " Width: ", self.WIDTH, " Empty tile: ", self.empty())
        print(self.tiles.reshape((self.WIDTH, self.WIDTH)))
        print("H1 cost: ", self.h1(), " H2 cost: ", self.h2(), " H3 cost: ", self.h3())
        print()


def test_run():
    # initialize the tree
    tree = Tree()

    # create the root node
    # size = int(input("Please enter the puzzle size (8, 15, 24...): "))
    # conf = input("Please enter the configuration of the puzzle,\n"
    #              "  using the format '1 2 3 4 5 6 7 8 b' where b=blank.\n"
    #              " You may type 'rand' to have the order generated for you: ")
    # type = input("Please select 'a' for A* search or 'b' for best-first search: ")
    size = 8
    conf = 'rand'
    search = 'a'

    root = Board(size, conf)
    tree.create_node(root.id(), root.id(), data=root)

    world = tree.get_node(root.id()).data

    # create nodes for all of the possibilities
    left = copy.copy(world)
    tree.create_node("left", "left", parent=world.id(), data=left)

    if search == 'a':
        print("A* search not implemented.")

    else:
        print("Best-first search not implemented.")

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


if __name__ == '__main__':

    test_run()
