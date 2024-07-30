import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random
import numpy as np
import time


class Node:

    def __init__(self):
        # track parent
        self.parent = None
        # track position
        self.i = 0
        self.j = 0
        #track costs
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')


def a_star(grid,start,goal):
    # need to define closed and open lists for nodes at every location
    rowsTotal = len(grid)
    colsTotal = len(grid[0])

    openList = []
    closedList = [[False for _ in range(colsTotal)] for _ in range(rowsTotal)]
    nodeInfo = [[Node() for _ in range(colsTotal)] for _ in range(rowsTotal)]

    animation_frames = []

    # start node 
    i, j = start
    nodeInfo[i][j].f = 0 
    nodeInfo[i][j].g = 0 
    nodeInfo[i][j].h = 0 
    heapq.heappush(openList,(0.0,i,j))

    # begin loop
    while openList:

        currentNodeInfo = heapq.heappop(openList) # get current node info
        i = currentNodeInfo[1] 
        j = currentNodeInfo[2]
        currentNode = nodeInfo[i][j] # get current node index
        currentNode.i = i
        currentNode.j = j
        closedList[i][j] = True # add current node to closed list since it is visited

        # store grid state for animation
        frame = [[0 if closedList[i][j] else (2 if grid[i][j] == 0 else 1) for j in range(colsTotal)] for i in range(rowsTotal)]
        frame[i][j] = 3  # current node
        animation_frames.append(frame)

        # goal check
        if (i, j) == (goal[0], goal[1]):
            #backtrack
            path = []
            current = currentNode
            while current.parent is not None:
                path.append((current.i, current.j))
                current = current.parent
            path.append((start[0], start[1])) # account for initial 
            # mark path for anim
            for (pi, pj) in path:
                animation_frames[-1][pi][pj] = 4
            return path[::-1], animation_frames

        # create kids

        directions = [(0,1),(0,-1),(1,0),(-1,0),
                      (1,1),(1,-1),(-1,1),(-1,-1)]
        
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            #check bounds
            if new_i > (len(grid)-1) or new_i < 0 or new_j < 0 or new_j > (len(grid[0])-1):
                continue

            #check terrain validity
            if grid[new_i][new_j] == 0:
                continue

            # check if child in closedList
            if closedList[new_i][new_j] == True:
                continue

            nodeDistance = math.sqrt((i-new_i)**2 + (j - new_j)**2)
            heuristic = math.sqrt((goal[0]-new_i)**2 + (goal[1] - new_j)**2)

            child = nodeInfo[new_i][new_j]
            new_g = currentNode.g + nodeDistance

            # update child cost if shorter path found
            if new_g < child.g:
                child.g = new_g
                child.h = heuristic
                child.f = child.g + child.h
                child.i = new_i
                child.j = new_j
                child.parent = currentNode

            # add child to open list
                heapq.heappush(openList, (child.f,new_i,new_j))


def generate_grid(rows, cols, obstacle_prob=0.2):

    grid = np.ones((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_prob:
                grid[i][j] = 0

    grid[49][0] = 1 # starting point, doesnt matter
    grid[0][49] = 1

    return grid


grid = generate_grid(50,50,obstacle_prob=0.33)

start = [49,0]
goal = [0,49]
start_time = time.time()
path, animation_frames = a_star(grid, start, goal)
end = time.time()

runtime = end-start_time

print("time = ", runtime, "seconds")

# animate
fig, ax = plt.subplots()
cmap = plt.get_cmap('tab10', 5)  # 5 colors for open, obstacle, closed, current, path

def update(frame):
    ax.clear()
    ax.imshow(frame, cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=animation_frames, repeat=False, interval = 0.001)
plt.show()

