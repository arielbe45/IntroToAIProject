import typing
from collections import deque

Node = typing.TypeVar('Node')


def bfs_shortest_paths(start_node: Node, get_free_neighbors: typing.Callable[[Node], typing.List[Node]]) \
        -> typing.Dict[Node, typing.List[Node]]:
    # Initialize a dictionary to store the shortest path for each node
    shortest_paths = {start_node: [start_node]}

    # Initialize a queue for BFS with the start node
    queue = deque([start_node])

    # While there are nodes to process
    while queue:
        current_node = queue.popleft()

        # Get neighbors of the current node
        neighbors = get_free_neighbors(current_node)

        for neighbor in neighbors:
            # If the neighbor hasn't been visited yet
            if neighbor not in shortest_paths:
                # Add it to the shortest paths dictionary with the updated path
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
                # Enqueue the neighbor for further exploration
                queue.append(neighbor)

    return shortest_paths


def bfs_distance_to_goal(start_node: Node, get_free_neighbors: typing.Callable[[Node], typing.List[Node]],
                         check_goal: typing.Callable[[Node], bool]) -> int:
    # Initialize a dictionary to store the distance to goal for each node
    distance_from_start = {start_node: 0}

    if check_goal(start_node):
        return 0

    # Initialize a queue for BFS with the start node
    queue = deque([start_node])

    # While there are nodes to process
    while queue:
        current_node = queue.popleft()
        if check_goal(current_node):
            return distance_from_start[current_node]

        # Get neighbors of the current node
        neighbors = get_free_neighbors(current_node)

        for neighbor in neighbors:
            # If the neighbor hasn't been visited yet
            if neighbor not in distance_from_start:
                # Add it to the distance_from_start dictionary with the updated distance
                distance_from_start[neighbor] = distance_from_start[current_node] + 1
                # Enqueue the neighbor for further exploration
                queue.append(neighbor)

    return float('inf')


def bfs_best_move(start_node: Node, get_legal_movements: typing.Callable[[Node], typing.Dict[typing.Any, Node]],
                  check_goal: typing.Callable[[Node], bool]):
    # Initialize a dictionary to store the distance to goal for each node
    distance_from_start = {start_node: 0}

    if check_goal(start_node):
        raise Exception

    # Initialize a queue for BFS with the start node
    queue = deque([(None, start_node)])

    # While there are nodes to process
    while queue:
        original_move, current_node = queue.popleft()
        if check_goal(current_node):
            return original_move

        # Get neighbors of the current node
        legal_movements = get_legal_movements(current_node)

        for move, neighbor in legal_movements:
            # If the neighbor hasn't been visited yet
            if neighbor not in distance_from_start:
                # Add it to the distance_from_start dictionary with the updated distance
                distance_from_start[neighbor] = distance_from_start[current_node] + 1
                # Enqueue the neighbor for further exploration
                if original_move is None:
                    original_move = move
                queue.append((original_move, neighbor))

    raise Exception
