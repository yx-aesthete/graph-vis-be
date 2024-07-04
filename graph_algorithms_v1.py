import random
from models import GraphDTO, NodeDTO, EdgeDTO


def generate_bfs_sequence_v1(graph: GraphDTO) -> list[int]:
    nodes = {node.id: node for node in graph.nodes}
    edges = graph.edges

    visited = set()
    queue = [graph.nodes[0].id]
    sequence = []

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            sequence.append(current)
            for edge in edges:
                if edge.source == current and edge.target not in visited:
                    queue.append(edge.target)

    return sequence


def generate_dfs_sequence_v1(graph: GraphDTO) -> list[int]:
    nodes = {node.id: node for node in graph.nodes}
    edges = graph.edges

    visited = set()
    stack = [graph.nodes[0].id]
    sequence = []

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            sequence.append(current)
            for edge in edges:
                if edge.source == current and edge.target not in visited:
                    stack.append(edge.target)

    return sequence


def generate_random_tree_graph(num_nodes: int) -> GraphDTO:
    nodes = [NodeDTO(id=i, x=0, y=0) for i in range(num_nodes)]
    edges = []

    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        edges.append(EdgeDTO(source=parent, target=i))

    return GraphDTO(nodes=nodes, edges=edges)
