# models.py
from pydantic import BaseModel
from typing import List


class NodeDTO(BaseModel):
    id: int
    x: float = 0.0
    y: float = 0.0


class EdgeDTO(BaseModel):
    source: int
    target: int


class GraphDTO(BaseModel):
    nodes: List[NodeDTO]
    edges: List[EdgeDTO]
