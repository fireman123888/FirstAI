from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum


class LayerType(str, Enum):
    INPUT = "input"
    CONV2D = "conv2d"
    LINEAR = "linear"
    MAXPOOL2D = "maxpool2d"
    BATCHNORM = "batchnorm"
    DROPOUT = "dropout"
    FLATTEN = "flatten"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    OUTPUT = "output"


class NodeInfo(BaseModel):
    id: str
    type: LayerType
    label: str
    params: Dict[str, Any]


class EdgeInfo(BaseModel):
    source: str
    target: str


class NetworkStructure(BaseModel):
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]


class TrainingConfig(BaseModel):
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    optimizer: str = "adam"


class TrainRequest(BaseModel):
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]
    config: TrainingConfig


class GenerateResponse(BaseModel):
    code: str


class TrainStartResponse(BaseModel):
    task_id: str
    message: str


class TrainStatusResponse(BaseModel):
    status: str
    current_epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    history: List[Dict[str, float]]
