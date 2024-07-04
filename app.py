from flask import Flask, request, jsonify
from flask_cors import CORS
from graph_algorithms import generate_traversal_path
import os
from graph_algorithms_v1 import (
    generate_bfs_sequence_v1,
    generate_dfs_sequence_v1,
    generate_random_tree_graph
)
from learn_graph import (
    generate_random_build_graph,
    get_graph_formats
)
from models import GraphDTO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

REACT_APP_API_URL = os.environ.get(
    "REACT_APP_API_URL", "http://127.0.0.1:5001")
port = int(os.environ.get("PORT", 5001))

# traversal first approach hard to delete but will do ultimately


@app.route("/traversal/bfs", methods=["POST"])
def get_bfs_traversal():
    try:
        graph_data = request.json
        graph = GraphDTO(**graph_data)
        bfs_sequence = generate_bfs_sequence_v1(graph)
        return jsonify(bfs_sequence)
    except Exception as e:
        print(f"Error in get_bfs_traversal: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/traversal/dfs", methods=["POST"])
def get_dfs_traversal():
    try:
        graph_data = request.json
        graph = GraphDTO(**graph_data)
        dfs_sequence = generate_dfs_sequence_v1(graph)
        return jsonify(dfs_sequence)
    except Exception as e:
        print(f"Error in get_dfs_traversal: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_random_tree_graph_for_traversal", methods=["POST"])
def generate_random_tree_graph_endpoint():
    try:
        num_nodes = request.json.get("num_nodes", 10)
        print(f"Received request to generate graph with {num_nodes} nodes")
        graph = generate_random_tree_graph(num_nodes)
        print(f"Generated graph: {graph}")
        return jsonify(graph.dict())
    except Exception as e:
        print(f"Error in generate_random_tree_graph: {e}")
        return jsonify({"error": str(e)}), 500


# the nice one starts here

@app.route("/generate_random_build_graph", methods=["POST"])
def generate_random_graph_endpoint():
    try:
        data = request.json
        num_nodes = data.get("num_nodes", 10)
        num_edges = data.get("num_edges", None)
        connectivity = data.get("connectivity", "random")
        additional_params = data.get("additional_params", {})
        print(f"Received request to generate graph with {num_nodes} nodes, {
              num_edges} edges, and {connectivity} connectivity")
        graph = generate_random_build_graph(
            num_nodes, num_edges, connectivity, **additional_params)
        graph_formats = get_graph_formats(graph)
        return jsonify({
            "graph": graph.dict(),
            "formats": graph_formats
        })
    except Exception as e:
        print(f"Error in generate_random_graph: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/traversal", methods=["POST"])
def get_traversal():
    try:
        data = request.json
        graph_data = data.get("graph")
        graph = GraphDTO(**graph_data)
        start_node = data.get("start_node")
        goal_node = data.get("goal_node", None)
        algorithm = data.get("algorithm")

        if not start_node or not algorithm:
            return jsonify({"error": "start_node and algorithm are required"}), 400

        traversal_sequence = generate_traversal_path(
            graph, start_node, algorithm, goal_node)
        return jsonify(traversal_sequence)
    except Exception as e:
        print(f"Error in get_traversal: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
