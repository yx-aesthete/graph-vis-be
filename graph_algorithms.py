# graph_algorithms.py
from collections import deque
import heapq
import networkx as nx


def generate_bfs_sequence(graph, start_node):
    adj_list = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        adj_list[edge.source].append(edge.target)
        adj_list[edge.target].append(edge.source)

    visited = set()
    queue = deque([start_node])
    sequence = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            sequence.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return sequence


def generate_dfs_sequence(graph, start_node):
    adj_list = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        adj_list[edge.source].append(edge.target)
        adj_list[edge.target].append(edge.source)

    visited = set()
    stack = [start_node]
    sequence = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            sequence.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return sequence


def generate_dijkstra_sequence(graph, start_node):
    adj_list = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        # Assuming weight is 1 for simplicity
        adj_list[edge.source].append((edge.target, 1))
        adj_list[edge.target].append((edge.source, 1))

    distances = {node.id: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    sequence = []

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue

        sequence.append(current_node)

        for neighbor, weight in adj_list[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return sequence


def generate_a_star_sequence(graph, start_node, goal_node):
    adj_list = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        # Assuming weight is 1 for simplicity
        adj_list[edge.source].append((edge.target, 1))
        adj_list[edge.target].append((edge.source, 1))

    def heuristic(node, goal):
        return abs(node - goal)  # Simplified heuristic for demonstration

    open_set = [(0, start_node)]
    came_from = {}
    g_score = {node.id: float('inf') for node in graph.nodes}
    g_score[start_node] = 0
    f_score = {node.id: float('inf') for node in graph.nodes}
    f_score[start_node] = heuristic(start_node, goal_node)
    sequence = []

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            break

        sequence.append(current)

        for neighbor, weight in adj_list[current]:
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + \
                    heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return sequence


def generate_min_cut_max_flow(graph, source, sink):
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.id)
    for edge in graph.edges:
        # Assuming capacity is 1 for simplicity
        G.add_edge(edge.source, edge.target, capacity=1)

    cut_value, partition = nx.minimum_cut(G, source, sink)
    reachable, non_reachable = partition
    min_cut_edges = [(u, v)
                     for u in reachable for v in G[u] if v in non_reachable]

    return {
        "cut_value": cut_value,
        "min_cut_edges": min_cut_edges
    }


ALGORITHMS = {
    "bfs": generate_bfs_sequence,
    "dfs": generate_dfs_sequence,
    "dijkstra": generate_dijkstra_sequence,
    "a_star": generate_a_star_sequence,
    "min_cut_max_flow": generate_min_cut_max_flow
}


def generate_traversal_path(graph, start_node, algorithm, goal_node=None):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm {algorithm} is not supported.")

    if algorithm in ["a_star", "min_cut_max_flow"] and goal_node is not None:
        return ALGORITHMS[algorithm](graph, start_node, goal_node)
    else:
        return ALGORITHMS[algorithm](graph, start_node)
