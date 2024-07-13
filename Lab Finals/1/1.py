a) BFS 
There are two methods- using deque or using heapq module. Using dequeu is more preferable.

(i) Using double ended queue
from collections import deque

def best_first_search_deque(graph, start, goal, heuristic):
    # Priority queue for exploring nodes based on heuristic + actual cost
    queue = deque([(start, heuristic[start], 0)])
    visited = set()
    parent = {start: None}
    cost = {start: 0}

    while queue:
        # Sort queue based on heuristic values + actual cost to simulate priority queue
        queue = deque(sorted(list(queue), key=lambda x: x[1] + x[2]))
        current_node, current_heuristic, current_cost = queue.popleft()

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            break

        for neighbor, edge_cost in graph[current_node]:
            if neighbor not in visited:
                new_cost = current_cost + edge_cost
                queue.append((neighbor, heuristic[neighbor], new_cost))
                parent[neighbor] = current_node
                cost[neighbor] = new_cost

    path = []
    node = goal
    total_cost = cost.get(goal, 0)
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, total_cost

# Example graph with manually assigned costs
graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 3), ('E', 4)],
    'C': [('F', 5), ('G', 6)],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

# Example heuristic values (assumed for demonstration)
heuristic = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 0,
    'E': 2,
    'F': 3,
    'G': 1
}

start = 'A'
goal = 'D'

path, total_cost = best_first_search_deque(graph, start, goal, heuristic)
print("Best First Search Path:", path)
print("Total Cost:", total_cost)


(ii) Using heapq

import heapq

def best_first_search(graph, start, goal, heuristic):
    # Priority queue for exploring nodes based on heuristic + actual cost
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start], start, 0))
    visited = set()
    parent = {start: None}
    cost = {start: 0}

    while priority_queue:
        current_heuristic, current_node, current_cost = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            break

        for neighbor, edge_cost in graph[current_node]:
            if neighbor not in visited:
                new_cost = current_cost + edge_cost
                heapq.heappush(priority_queue, (heuristic[neighbor] + new_cost, neighbor, new_cost))
                parent[neighbor] = current_node
                cost[neighbor] = new_cost

    path = []
    node = goal
    total_cost = cost.get(goal, 0)
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, total_cost

# Example graph with manually assigned costs
graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 3), ('E', 4)],
    'C': [('F', 5), ('G', 6)],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

# Example heuristic values (assumed for demonstration)
heuristic = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 0,
    'E': 2,
    'F': 3,
    'G': 1
}

start = 'A'
goal = 'D'

path, total_cost = best_first_search(graph, start, goal, heuristic)
print("Best First Search Path:", path)
print("Total Cost:", total_cost)



b) 3-D surface

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Doors']
z = dataset['Price']

ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap="jet")
ax.set_title("3D Surface Plot")

plt.show()
