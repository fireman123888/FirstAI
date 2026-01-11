import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type Connection,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type NodeChange,
  type EdgeChange,
} from '@xyflow/react';
import type { LayerNodeData, TrainingConfig, TrainingStatus } from '../types';

interface AppState {
  // 节点和边
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
  selectedNode: Node<LayerNodeData> | null;

  // 生成的代码
  generatedCode: string;

  // 训练相关
  trainingConfig: TrainingConfig;
  trainingStatus: TrainingStatus;

  // Actions
  setNodes: (nodes: Node<LayerNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (node: Node<LayerNodeData>) => void;
  updateNodeData: (nodeId: string, data: Partial<LayerNodeData>) => void;
  setSelectedNode: (node: Node<LayerNodeData> | null) => void;
  setGeneratedCode: (code: string) => void;
  setTrainingConfig: (config: Partial<TrainingConfig>) => void;
  setTrainingStatus: (status: Partial<TrainingStatus>) => void;
  clearCanvas: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNode: null,
  generatedCode: '',

  trainingConfig: {
    epochs: 10,
    learning_rate: 0.001,
    batch_size: 64,
    optimizer: 'adam',
  },

  trainingStatus: {
    status: 'idle',
    current_epoch: 0,
    total_epochs: 0,
    loss: 0,
    accuracy: 0,
    history: [],
  },

  setNodes: (nodes) => set({ nodes }),

  setEdges: (edges) => set({ edges }),

  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes) as Node<LayerNodeData>[],
    });
  },

  onEdgesChange: (changes) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },

  onConnect: (connection) => {
    set({
      edges: addEdge({ ...connection, animated: true }, get().edges),
    });
  },

  addNode: (node) => {
    set({
      nodes: [...get().nodes, node],
    });
  },

  updateNodeData: (nodeId, data) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      ),
    });
  },

  setSelectedNode: (node) => set({ selectedNode: node }),

  setGeneratedCode: (code) => set({ generatedCode: code }),

  setTrainingConfig: (config) =>
    set({ trainingConfig: { ...get().trainingConfig, ...config } }),

  setTrainingStatus: (status) =>
    set({ trainingStatus: { ...get().trainingStatus, ...status } }),

  clearCanvas: () => set({ nodes: [], edges: [], selectedNode: null }),
}));
