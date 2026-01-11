import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Callable, Optional
import uuid
import asyncio
import base64
import io
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from ..models.schemas import NodeInfo, EdgeInfo, TrainingConfig

# 存储训练任务状态
training_tasks: Dict[str, Dict[str, Any]] = {}

executor = ThreadPoolExecutor(max_workers=2)


def build_model_from_structure(nodes: List[NodeInfo], edges: List[EdgeInfo]) -> nn.Module:
    """根据网络结构构建 PyTorch 模型"""

    # 构建节点顺序
    node_map = {n.id: n for n in nodes}
    target_nodes = {e.target for e in edges}
    start_nodes = [n for n in nodes if n.id not in target_nodes]

    adjacency = {}
    for edge in edges:
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        adjacency[edge.source].append(edge.target)

    ordered_nodes = []
    visited = set()
    queue = [n.id for n in start_nodes]

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        ordered_nodes.append(node_map[node_id])
        if node_id in adjacency:
            for next_id in adjacency[node_id]:
                if next_id not in visited:
                    queue.append(next_id)

    # 构建层列表
    layers = []
    for node in ordered_nodes:
        layer = create_layer(node)
        if layer is not None:
            layers.append(layer)

    class DynamicModel(nn.Module):
        def __init__(self, layer_list):
            super().__init__()
            self.layers = nn.ModuleList([l for l in layer_list if isinstance(l, nn.Module)])
            self.all_layers = layer_list

        def forward(self, x):
            for layer in self.all_layers:
                if isinstance(layer, str):
                    if layer == "flatten":
                        x = x.view(x.size(0), -1)
                    elif layer == "relu":
                        x = F.relu(x)
                    elif layer == "sigmoid":
                        x = torch.sigmoid(x)
                    elif layer == "softmax":
                        x = F.softmax(x, dim=1)
                else:
                    x = layer(x)
            return x

    return DynamicModel(layers)


def create_layer(node: NodeInfo):
    """根据节点类型创建对应的 PyTorch 层"""
    p = node.params
    layer_type = node.type.value

    if layer_type == "input" or layer_type == "output":
        return None
    elif layer_type == "conv2d":
        return nn.Conv2d(
            in_channels=p.get("in_channels", 1),
            out_channels=p.get("out_channels", 32),
            kernel_size=p.get("kernel_size", 3),
            stride=p.get("stride", 1),
            padding=p.get("padding", 0)
        )
    elif layer_type == "linear":
        return nn.Linear(
            in_features=p.get("in_features", 128),
            out_features=p.get("out_features", 64)
        )
    elif layer_type == "maxpool2d":
        return nn.MaxPool2d(
            kernel_size=p.get("kernel_size", 2),
            stride=p.get("stride", 2)
        )
    elif layer_type == "batchnorm":
        return nn.BatchNorm2d(num_features=p.get("num_features", 32))
    elif layer_type == "dropout":
        return nn.Dropout(p=p.get("p", 0.5))
    elif layer_type == "flatten":
        return "flatten"
    elif layer_type == "relu":
        return "relu"
    elif layer_type == "sigmoid":
        return "sigmoid"
    elif layer_type == "softmax":
        return "softmax"

    return None


def run_training(
    task_id: str,
    nodes: List[NodeInfo],
    edges: List[EdgeInfo],
    config: TrainingConfig
):
    """执行训练任务"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 更新任务状态
    training_tasks[task_id] = {
        "status": "running",
        "current_epoch": 0,
        "total_epochs": config.epochs,
        "loss": 0.0,
        "accuracy": 0.0,
        "history": [],
        "nodes": nodes,  # 保存网络结构用于后续预测
        "edges": edges,
    }

    try:
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载 MNIST 数据集
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # 构建模型
        model = build_model_from_structure(nodes, edges).to(device)

        # 优化器
        if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

        # 训练循环
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count

            # 评估
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            accuracy = correct / total

            # 更新状态
            training_tasks[task_id].update({
                "current_epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
            })
            training_tasks[task_id]["history"].append({
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy
            })

        # 保存模型
        torch.save(model.state_dict(), f"models/{task_id}.pth")

        training_tasks[task_id]["status"] = "completed"

    except Exception as e:
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["error"] = str(e)


async def start_training_task(
    nodes: List[NodeInfo],
    edges: List[EdgeInfo],
    config: TrainingConfig
) -> str:
    """启动训练任务"""
    task_id = str(uuid.uuid4())

    # 在线程池中执行训练
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, run_training, task_id, nodes, edges, config)

    return task_id


def get_training_status(task_id: str) -> Optional[Dict[str, Any]]:
    """获取训练状态"""
    return training_tasks.get(task_id)


def get_sample_predictions(task_id: str, count: int = 6) -> List[Dict[str, Any]]:
    """获取模型预测示例"""
    task = training_tasks.get(task_id)
    if task is None:
        raise ValueError("任务不存在")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重建模型
    nodes = task["nodes"]
    edges = task["edges"]
    model = build_model_from_structure(nodes, edges).to(device)

    # 加载训练好的权重
    model.load_state_dict(torch.load(f"models/{task_id}.pth", map_location=device))
    model.eval()

    # 加载测试数据（不做归一化，保留原图用于显示）
    test_dataset = datasets.MNIST('./data', train=False, download=True)

    # 用于模型推理的 transform
    inference_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 随机选择样本
    indices = random.sample(range(len(test_dataset)), count)
    results = []

    with torch.no_grad():
        for idx in indices:
            # 获取原始图像和标签
            img, label = test_dataset[idx]

            # 将图像转为 base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 进行预测
            img_tensor = inference_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probabilities[0][pred].item()

            results.append({
                "image": img_base64,
                "label": label,
                "prediction": pred,
                "correct": label == pred,
                "confidence": round(confidence * 100, 1)
            })

    return results
