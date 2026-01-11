import os
import anthropic
from typing import List, Dict, Any
from ..models.schemas import NodeInfo, EdgeInfo


def build_model_description(nodes: List[NodeInfo], edges: List[EdgeInfo]) -> str:
    """将节点和边转换为模型描述文本"""

    # 构建节点顺序（拓扑排序）
    node_map = {n.id: n for n in nodes}

    # 找到输入节点
    target_nodes = {e.target for e in edges}
    source_nodes = {e.source for e in edges}

    # 输入节点：没有入边的节点
    start_nodes = [n for n in nodes if n.id not in target_nodes]

    # 构建邻接表
    adjacency = {}
    for edge in edges:
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        adjacency[edge.source].append(edge.target)

    # BFS 遍历获取顺序
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

    # 构建描述
    layers_desc = []
    for i, node in enumerate(ordered_nodes):
        layer_desc = f"{i+1}. {node.label} ({node.type.value})"
        if node.params:
            params_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
            layer_desc += f": {params_str}"
        layers_desc.append(layer_desc)

    return "\n".join(layers_desc)


def get_anthropic_client():
    """获取 Anthropic 客户端，支持多种配置方式"""

    # 支持多种环境变量名称
    api_key = (
        os.getenv("ANTHROPIC_API_KEY") or
        os.getenv("ANTHROPIC_AUTH_TOKEN") or
        os.getenv("CLAUDE_API_KEY")
    )

    if not api_key:
        return None

    # 支持自定义 base_url（用于代理或私有部署）
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    if base_url:
        return anthropic.Anthropic(api_key=api_key, base_url=base_url)
    else:
        return anthropic.Anthropic(api_key=api_key)


async def generate_pytorch_code(nodes: List[NodeInfo], edges: List[EdgeInfo]) -> str:
    """使用 Claude API 生成 PyTorch 代码"""

    model_desc = build_model_description(nodes, edges)

    prompt = f"""请根据以下神经网络结构，生成完整的 PyTorch 代码。

网络结构（按层顺序）：
{model_desc}

要求：
1. 生成一个完整的 PyTorch 神经网络类（继承 nn.Module）
2. 包含完整的 __init__ 和 forward 方法
3. 添加训练函数和评估函数
4. 使用 MNIST 数据集作为示例
5. 代码要可以直接运行
6. 包含适当的中文注释

只输出 Python 代码，不要其他解释。代码要完整且可运行。"""

    client = get_anthropic_client()

    if not client:
        # 如果没有配置 API，返回模板代码
        return generate_template_code(nodes, edges)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        code = message.content[0].text

        # 清理代码（移除 markdown 代码块标记）
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    except Exception as e:
        print(f"AI 生成失败: {e}")
        # 出错时回退到模板代码
        return generate_template_code(nodes, edges)


def generate_template_code(nodes: List[NodeInfo], edges: List[EdgeInfo]) -> str:
    """生成模板代码（无 API 时的备用方案）"""

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

    # 生成层定义代码
    layer_defs = []
    forward_calls = []

    for i, node in enumerate(ordered_nodes):
        layer_name = f"layer_{i}"

        if node.type.value == "input":
            continue
        elif node.type.value == "conv2d":
            p = node.params
            layer_defs.append(
                f"        self.{layer_name} = nn.Conv2d({p.get('in_channels', 1)}, "
                f"{p.get('out_channels', 32)}, kernel_size={p.get('kernel_size', 3)}, "
                f"stride={p.get('stride', 1)}, padding={p.get('padding', 0)})"
            )
            forward_calls.append(f"        x = self.{layer_name}(x)")
        elif node.type.value == "linear":
            p = node.params
            layer_defs.append(
                f"        self.{layer_name} = nn.Linear({p.get('in_features', 128)}, "
                f"{p.get('out_features', 64)})"
            )
            forward_calls.append(f"        x = self.{layer_name}(x)")
        elif node.type.value == "maxpool2d":
            p = node.params
            layer_defs.append(
                f"        self.{layer_name} = nn.MaxPool2d(kernel_size={p.get('kernel_size', 2)}, "
                f"stride={p.get('stride', 2)})"
            )
            forward_calls.append(f"        x = self.{layer_name}(x)")
        elif node.type.value == "batchnorm":
            p = node.params
            layer_defs.append(
                f"        self.{layer_name} = nn.BatchNorm2d({p.get('num_features', 32)})"
            )
            forward_calls.append(f"        x = self.{layer_name}(x)")
        elif node.type.value == "dropout":
            p = node.params
            layer_defs.append(
                f"        self.{layer_name} = nn.Dropout(p={p.get('p', 0.5)})"
            )
            forward_calls.append(f"        x = self.{layer_name}(x)")
        elif node.type.value == "flatten":
            forward_calls.append("        x = x.view(x.size(0), -1)")
        elif node.type.value == "relu":
            forward_calls.append("        x = F.relu(x)")
        elif node.type.value == "sigmoid":
            forward_calls.append("        x = torch.sigmoid(x)")
        elif node.type.value == "softmax":
            forward_calls.append("        x = F.softmax(x, dim=1)")
        elif node.type.value == "output":
            pass

    layers_code = "\n".join(layer_defs) if layer_defs else "        pass"
    forward_code = "\n".join(forward_calls) if forward_calls else "        pass"

    template = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
{layers_code}

    def forward(self, x):
{forward_code}
        return x


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {{epoch}} [{{batch_idx * len(data)}}/{{len(train_loader.dataset)}}] Loss: {{loss.item():.6f}}')


# 评估函数
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {{test_loss:.4f}}, Accuracy: {{correct}}/{{len(test_loader.dataset)}} ({{accuracy:.2f}}%)')
    return test_loss, accuracy


# 主函数
def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {{device}}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 创建模型
    model = NeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    main()
'''

    return template
