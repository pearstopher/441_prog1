# CS441 Programming Assignment 1
# Christopher Juncker

import numpy as np
import math
import copy
from treelib import Tree, exceptions as x  # pip install treelib
import heapq as hq  # priority queue

# todo: implement 'test suite' to run all heuristics for each search method


class Board:

    def __init__(self, size, conf="rand"):
        self.SIZE = size + 1  # size = total number of spaces on the board
        self.WIDTH = int(math.sqrt(self.SIZE))
        self.parent = None

        if conf == "rand":
            # generate n squares at random locations
            self.tiles = np.random.choice(self.SIZE, self.SIZE, replace=False)
            while not self.solvable():
                self.tiles = np.random.choice(self.SIZE, self.SIZE, replace=False)

        else:
            # I am using 0 to represent the empty square
            conf = conf.replace("b", "0")
            # split the string and build the array
            self.tiles = np.array(list(map(int, conf.split())))

        self.goal_tiles = np.array(list(range(0, self.SIZE)))
        # print(self.goal)
        # print(self.tiles)

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
        # print(cost)
        return cost

    # heuristic 2 - total Euclidean distance (L2 norm)
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

    # DISPLAY & INFO

    # return unique identifier for this board configuration
    def id(self):
        # convert board configuration into a unique integer
        id_str = ""
        for i in self.tiles:
            id_str += str(i)
        return int(id_str)

    def solvable(self):
        # test if board is solvable by looking for out-of-order elements
        total = 0
        for i in range(0, self.SIZE - 1):
            for j in range(i + 1, self.SIZE):
                if self.tiles[i] != 0 and self.tiles[j] != 0 and self.tiles[i] > self.tiles[j]:
                    total += 1
        return total % 2 == 0

    def goal(self):
        # check if this board has reached the goal
        # print(self.tiles)
        # print(self.goal_tiles)
        # print()
        if np.array_equal(self.tiles, self.goal_tiles):
            return True
        return False

    # print some basic info about the board
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
            self.heuristic = input("Please select a heuristic (1, 2, 3): ")

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
        hq.heappush(self.pq, self.PQNode(root.h3(), root.id()))

    def expand(self, node):
        # generate costs for the current node and add to tree and priority queue
        if node.goal():
            self.found = node

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
                # get the cost from specified heuristic
                if self.heuristic == 1:
                    cost = c.h1()
                elif self.heuristic == 2:
                    cost = c.h2()
                else:
                    cost = c.h3()
                # add depth to cost if the search is A*
                if self.type == 'a':
                    n = self.tree.get_node(c.id())
                    d = self.tree.depth(n)
                    cost += d
                hq.heappush(self.pq, self.PQNode(cost, c.id()))

    def expand_cheapest(self):
        cheapest = hq.heappop(self.pq)
        # cheapest = pq[0]
        node = self.tree.get_node(cheapest.id).data
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
    run.run(10000000)


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
    presets = ["b 1 2 3 5 6 4 7 8 9 10 11 12 13 14 15",
               "b 1 2 3 12 5 6 7 4 9 10 11 8 13 14 15",
               "3 2 1 b 6 5 4 7 8 9 10 11 12 13 14 15",
               "b 1 2 3 5 4 6 7 9 8 10 11 12 13 14 15",
               "b 1 2 3 6 5 4 7 10 9 8 11 12 13 14 15"]

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
    run = input("... ")

    if run == "2":
        run_assignment()
    elif run == "3":
        run_extra_credit()
    else:
        run_input()
