import { useCallback, useRef, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  type Node,
  type Connection,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { message } from 'antd';
import { useAppStore } from '../../store/useAppStore';
import type { LayerType, LayerNodeData, LayerParams } from '../../types';
import LayerNode from '../../nodes/LayerNode';
import './Canvas.css';

const nodeTypes = {
  layerNode: LayerNode,
};

// 默认参数
const defaultParams: Record<LayerType, LayerParams> = {
  input: { channels: 1, height: 28, width: 28 },
  conv2d: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
  linear: { in_features: 128, out_features: 64 },
  maxpool2d: { kernel_size: 2, stride: 2 },
  batchnorm: { num_features: 32 },
  dropout: { p: 0.5 },
  flatten: {},
  relu: {},
  sigmoid: {},
  softmax: {},
  output: { num_classes: 10 },
};

const layerLabels: Record<LayerType, string> = {
  input: 'Input',
  conv2d: 'Conv2D',
  linear: 'Linear',
  maxpool2d: 'MaxPool2D',
  batchnorm: 'BatchNorm',
  dropout: 'Dropout',
  flatten: 'Flatten',
  relu: 'ReLU',
  sigmoid: 'Sigmoid',
  softmax: 'Softmax',
  output: 'Output',
};

let nodeId = 0;
const getId = () => `node_${nodeId++}`;

function CanvasInner() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    setSelectedNode,
  } = useAppStore();

  // 验证连接是否有效
  const isValidConnection = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);

      if (!sourceNode || !targetNode) return false;

      const sourceType = (sourceNode.data as LayerNodeData).layerType;
      const targetType = (targetNode.data as LayerNodeData).layerType;

      // Output 不能作为源（不能连接到其他节点）
      if (sourceType === 'output') {
        message.warning('Output 层不能连接到其他层');
        return false;
      }

      // Input 不能作为目标（不能被其他节点连接）
      if (targetType === 'input') {
        message.warning('不能连接到 Input 层');
        return false;
      }

      // 检查是否已存在相同连接
      const existingEdge = edges.find(
        (e) => e.source === connection.source && e.target === connection.target
      );
      if (existingEdge) {
        message.warning('连接已存在');
        return false;
      }

      // 检查目标节点是否已有入边（简单网络只允许单入边）
      const hasIncomingEdge = edges.some((e) => e.target === connection.target);
      if (hasIncomingEdge) {
        message.warning('该节点已有输入连接');
        return false;
      }

      return true;
    },
    [nodes, edges]
  );

  const handleConnect = useCallback(
    (connection: Connection) => {
      if (isValidConnection(connection)) {
        onConnect(connection);
      }
    },
    [isValidConnection, onConnect]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow') as LayerType;

      if (!type || !reactFlowWrapper.current) {
        return;
      }

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = {
        x: event.clientX - bounds.left - 75,
        y: event.clientY - bounds.top - 25,
      };

      const newNode: Node<LayerNodeData> = {
        id: getId(),
        type: 'layerNode',
        position,
        data: {
          label: layerLabels[type],
          layerType: type,
          params: { ...defaultParams[type] },
        },
      };

      addNode(newNode);
    },
    [addNode]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node as Node<LayerNodeData>);
    },
    [setSelectedNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  return (
    <div className="canvas-wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        defaultEdgeOptions={{
          animated: true,
          style: { strokeWidth: 2 },
        }}
      >
        <Background gap={15} />
        <Controls />
        <MiniMap
          nodeStrokeWidth={3}
          zoomable
          pannable
          style={{ background: '#f5f5f5' }}
        />
      </ReactFlow>
    </div>
  );
}

export default function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
