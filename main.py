# CS441-003 Programming Assignment 1
# Christopher Juncker
#
# Instructions:
#   I had to 'pip install treelib' to get my tree going.
#   Everything else should work without any problems.
#   Tested on my Windows PC using Python 3.9 and on the
#     school servers using Python 3.8 without any issues.

import numpy as np
import math
import copy
from treelib import Tree, exceptions as x  # pip install treelib
import heapq as hq  # priority queue

"""
Board
    Class to hold board configurations. Each node in my tree contains one board object.
    
Variables:
    SIZE
        number of squares on the board. An n-puzzle board has n+1 squares
    WIDTH
        is equal to height because the board is square. WIDTH = sqrt(SIZE)
    tiles
        a 1-d array of tiles representing the current board configuration.
    goal_tiles
        a 1-d array of tiles representing the goal state or solved board.
    parent
        If assigned, a reference to the board used to generated this position.
        
Functions:
    init
        Set up initial board state information.
        
    [MOVEMENT]
    find(array, tile)
        Find the location of a tile in a given board array.
    empty
        Return the location of the empty square
    swap(tile1, tile2)
        Swap any two squares in the array. Only checks that indexes are valid.
    up
        Swaps the blank with the tile above it.
    down
        Swaps the blank with the tile below it.
    left
        Swaps the blank with the tile to it's left.
    right
        Swaps the blank with the tile on it's right.
        
    [HEURISTICS]
    row_diff(tile1, tile2)
        Returns how many rows away one tile is from another
    col_diff(tile1, tile2)
        Returns how many columns away one tile is from another
    h1
        First heuristic. Calculates number of tiles out of place.
    h2
        Second heuristic. Calculates straight line distance between tiles and
        their correct locations (Euclidean distance). Dominates h1.
    h3
        Third heuristic. Calculates number of swaps required to get each tile
        to it's correct location (Manhattan distance). Dominates h2.
    
    [BOARD INFO]
    id
        Returns an identifier which is unique for this board configuration.
    solvable
        Returns true if the board is has an even number of inversions. The
        rest of this program supports custom goal states so this will need
        to be modified to determine whether the goal state has an even or
        odd number of inversions.
    goal
        Returns true if the board state matches the goal state.
    info
        Prints information about the current board state. Not currently used.
        
"""


class Board:

    def __init__(self, size, conf="rand"):
        self.SIZE = size + 1  # size = total number of spaces on the board
        self.WIDTH = int(math.sqrt(self.SIZE))
        self.parent = None

        if conf == "rand":
            # generate n squares at random locations
            while not self.solvable():
                self.tiles = np.random.choice(self.SIZE, self.SIZE, replace=False)
        else:
            # I am using 0 to represent the empty square going forwards
            conf = conf.replace("b", "0")
            # split the configuration string and build the array
            self.tiles = np.array(list(map(int, conf.split())))

        # set the goal configuration
        self.goal_tiles = np.array(list(range(1, self.SIZE + 1)))
        # set last square to 0/blank
        self.goal_tiles[self.SIZE - 1] = 0
        # looking back, it was definitely a waste of space to have each board
        #   hold a goal state when the goal state never changes.

    # MOVEMENT FUNCTIONS
    @staticmethod
    def find(array, tile):
        return int(np.where(array == tile)[0])

    def empty(self):
        return self.find(self.tiles, 0)

    def swap(self, a, b):
        # check for invalid indexes
        if a < 0 or a > self.SIZE - 1 \
                or b < 0 or b > self.SIZE - 1:
            return False
        temp = self.tiles[a]
        self.tiles[a] = self.tiles[b]
        self.tiles[b] = temp
        return True

    def up(self):
        e = self.empty()
        # make sure there is a row above
        if e < self.WIDTH:
            return False
        self.swap(e, e - self.WIDTH)
        return True

    def down(self):
        e = self.empty()
        # make sure there is a row below
        if e > self.SIZE - 1 - self.WIDTH:
            return False
        self.swap(e, e + self.WIDTH)
        return True

    def left(self):
        e = self.empty()
        # make sure there is a column to the left
        if e % self.WIDTH == 0:
            return False
        self.swap(e, e - 1)
        return True

    def right(self):
        e = self.empty()
        # make sure there is a column to the right
        if (e + 1) % self.WIDTH == 0:
            return False
        self.swap(e, e + 1)
        return True

    # HEURISTIC FUNCTIONS

    def row_diff(self, num):
        a = self.find(self.tiles, num)
        b = self.find(self.goal_tiles, num)
        # diff = abs(math.floor(a / self.WIDTH) - math.floor(b / self.WIDTH))
        diff = abs((a // self.WIDTH) - (b // self.WIDTH))  # just learned about the //
        return diff

    def col_diff(self, num):
        a = self.find(self.tiles, num)
        b = self.find(self.goal_tiles, num)
        diff = abs((a % self.WIDTH) - (b % self.WIDTH))
        return diff

    # heuristic 1 - number of misplaced tiles.
    def h1(self):
        cost = 0
        for i in range(1, self.SIZE):
            if self.row_diff(i) or self.col_diff(i):
                cost += 1
        return cost

    # heuristic 2 - total Euclidean distance (L2 norm). Dominates h1.
    def h2(self):
        cost = 0
        for i in range(1, self.SIZE):
            cost += math.sqrt(pow(self.row_diff(i), 2) + pow(self.col_diff(i), 2))
        return cost

    # heuristic 3 - total Manhattan distance (L1 norm). Dominates h2.
    def h3(self):
        cost = 0
        for i in range(1, self.SIZE):
            cost += self.row_diff(i) + self.col_diff(i)
        return cost

    # INFO FUNCTIONS

    def id(self):
        # convert board configuration into a unique string
        id_str = ""
        for i in self.tiles:
            id_str += str(i).zfill(2)  # supports unique ids up to 10x10 board
        return id_str

    def solvable(self):
        if not self.tiles:
            return False
        # count the number of inversions
        total = 0
        for i in range(0, self.SIZE - 1):
            for j in range(i + 1, self.SIZE):
                if self.tiles[i] != 0 and self.tiles[j] != 0 and self.tiles[i] > self.tiles[j]:
                    total += 1
        # as mentioned earlier, in order to fully support custom goal states,
        #   I will need to also count the number of inversions in the goal.
        return total % 2 == 0

    def goal(self):
        if np.array_equal(self.tiles, self.goal_tiles):
            return True
        return False

    def info(self):
        print("Board ID: ", self.id())
        print("Size: ", self.SIZE, " Width: ", self.WIDTH, " Empty tile: ", self.empty())
        print(self.tiles.reshape((self.WIDTH, self.WIDTH)))
        print("H1 cost: ", self.h1(), " H2 cost: ", self.h2(), " H3 cost: ", self.h3())
        print("Solvable: ", self.solvable())
        print()


class TestRun:

    def __init__(self):
        # initialize the tree
        self.tree = Tree()
        # default info to start
        self.size = 8  # does this mess up goal configuration?
        # self.size = 15
        self.conf = "rand"
        self.type = 'a'
        self.heuristic = 1

        # set up a flag for if answer is found
        self.found = None
        # initialize priority queue
        self.pq = []

    # make a little node to hold our data in the priority q
    class PQNode:
        def __init__(self, cost, node_id):
            self.cost = cost
            self.id = node_id

        def __lt__(self, other):
            return self.cost < other.cost

    def configure(self, mode="default", c_size=None, c_conf=None, c_type=None, c_heuristic=None):

        if mode == "input":
            # create the root node
            self.size = int(input("Please enter the puzzle size (8, 15, 24...): "))
            self.conf = input("Please enter the configuration of the puzzle,\n"
                              "  using the format '1 2 3 4 5 6 7 8 b' where b=blank.\n"
                              " You may type 'rand' to have the order generated for you: ")
            self.type = input("Please select 'a' for A* search or 'b' for best-first search: ")
            self.heuristic = int(input("Please select a heuristic (1, 2, 3): "))

        if mode == "preset":
            self.size = c_size
            self.conf = c_conf
            self.type = c_type
            self.heuristic = c_heuristic

        # ^otherwise the program just uses the default values from init

        # create the root node based on the specifications
        root = Board(self.size, self.conf)
        # root.info()  # good information, keep silent for now
        self.tree.create_node(root.id(), root.id(), data=root)

        # push initial node onto the priority queue
        # hq.heappush(pq, [root.h3(), root.id()])
        hq.heappush(self.pq, self.PQNode(0, root.id()))

    def expand(self, node):
        # generate costs for the current node and add to tree and priority queue
        # if node.goal():
        #    # this may be a premature celebration
        #    # I am concerned that I have happened to find a solution
        #    # which is not the best solution
        #    # self.found = node

        for i in range(4):
            c = copy.deepcopy(node)
            c.parent = node
            # wtb switch statement
            if i == 0:
                c.left()
            elif i == 1:
                c.right()
            elif i == 2:
                c.up()
            elif i == 3:
                c.down()
            # ignore if resulting board is identical
            if c.id() == node.id():
                continue

            # otherwise add it to tree
            fail = False
            try:
                self.tree.create_node(c.id(), c.id(), parent=node.id(), data=c)
            except x.DuplicatedNodeIdError:
                # it could be a duplicate node. won't get picked of course but ID is in use
                fail = True

            # otherwise add to heap
            if not fail:
                cost = 0
                # get the cost from specified heuristic
                if self.heuristic == 1:
                    cost = c.h1()
                elif self.heuristic == 2:
                    cost = c.h2()
                elif self.heuristic == 3:
                    cost = c.h3()
                # print(cost, end=", ")

                # if cost == 0:
                #    self.found = c

                # add depth to cost if the search is A*
                if self.type == 'a':
                    n = self.tree.get_node(c.id())
                    d = self.tree.depth(n)
                    cost += d
                    # print(cost)
                hq.heappush(self.pq, self.PQNode(cost, c.id()))

    def expand_cheapest(self):
        cheapest = hq.heappop(self.pq)
        # cheapest = pq[0]
        node = self.tree.get_node(cheapest.id).data

        # check for a goal in cheapest node
        # instead of checking for goal in all expanded nodes (False goals)
        if node.goal():
            self.found = node
        else:
            self.expand(node)
        # print(pq)

    def run(self, limit=100000):
        for i in range(limit):
            self.expand_cheapest()
            # print("i: ", i, " cost: ", cost)
            if self.found:
                print("Nodes expanded: ", i)
                path = []
                path = self.show_path(self.found, path)
                print("Steps to solution: ", len(path))
                for j in range(len(path)):
                    # print(j.reshape(self.found.WIDTH, self.found.WIDTH))
                    print(path[j], end='')
                    if j != len(path) - 1:
                        print(" -> ", end='')
                print()
                break

    def show_tree(self):
        self.tree.show()

    def show_path(self, node, result):
        # result = str(node.id()).zfill(node.SIZE) + "\n" + result
        result.insert(0, node.tiles)
        if node.parent:
            return self.show_path(node.parent, result)
        else:
            return result


def run_default():
    run = TestRun()
    run.configure()
    run.run()


def run_input():
    run = TestRun()
    run.configure("input")
    run.run(100000)


def run_assignment():
    presets = ["7 8 b 6 2 3 5 4 1",
               "3 6 1 8 7 4 2 5 b",
               "3 b 8 6 7 1 2 4 5",
               "3 5 4 1 2 b 7 6 8",
               "4 2 6 3 b 7 8 1 5"]

    # for each search,
    for i in ("a", "b"):
        # for each heuristic,
        for j in range(1, 4):
            # for each preset
            for k in range(len(presets)):
                run = TestRun()
                s = "A*" if i == "a" else "Best-First"
                print("\nSEARCH", s, ",\tHEURISTIC", j, ",\tPRESET", k + 1, "(", presets[k], ")")
                run.configure("preset", 8, presets[k], i, j)
                run.run()


def run_extra_credit():
    presets = ["1 2 3 4 5 6 7 8 9 12 10 11 13 14 15 b",
               "1 2 3 12 5 6 7 4 9 10 11 8 13 14 15 b",
               "1 2 3 4 5 6 7 8 9 10 12 11 b 13 15 14",
               "1 2 3 4 5 6 7 8 9 10 12 11 13 15 14 b",
               "1 2 3 4 5 6 7 8 9 10 11 14 12 13 15 b"]

    # for each search,
    for i in ("a", "b"):
        # for each heuristic,
        for j in range(1, 4):
            # for each preset
            for k in range(len(presets)):
                run = TestRun()
                s = "A*" if i == "a" else "Best-First"
                print("\nSEARCH", s, ",\tHEURISTIC", j, ",\tPRESET", k + 1, "(", presets[k], ")")
                run.configure("preset", 15, presets[k], i, j)
                run.run(10000000)  # 100k default is nothing!


if __name__ == '__main__':

    print("Please make a selection:")
    print("\t1) User Input")
    print("\t2) Assignment")
    print("\t3) Extra Credit")
    run_mode = input("... ")

    if run_mode == "2":
        run_assignment()
    elif run_mode == "3":
        run_extra_credit()
    else:
        run_input()
