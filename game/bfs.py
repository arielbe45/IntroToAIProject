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
