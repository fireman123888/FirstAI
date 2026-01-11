import axios from 'axios';
import type { Node, Edge } from '@xyflow/react';
import type { LayerNodeData, TrainingConfig } from '../types';

// 支持环境变量配置，默认使用本地地址
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api';
export const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8002/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export async function generateCode(
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): Promise<string> {
  const response = await api.post('/generate', {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.layerType,
      label: n.data.label,
      params: n.data.params,
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
    })),
  });
  return response.data.code;
}

export async function startTraining(
  nodes: Node<LayerNodeData>[],
  edges: Edge[],
  config: TrainingConfig
): Promise<{ task_id: string }> {
  const response = await api.post('/train/start', {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.layerType,
      label: n.data.label,
      params: n.data.params,
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
    })),
    config,
  });
  return response.data;
}

export async function getTrainingStatus(taskId: string) {
  const response = await api.get(`/train/${taskId}`);
  return response.data;
}

export interface PredictionResult {
  image: string;
  label: number;
  prediction: number;
  correct: boolean;
  confidence: number;
}

export async function getPredictions(taskId: string, count: number = 6): Promise<PredictionResult[]> {
  const response = await api.get(`/train/${taskId}/predictions`, {
    params: { count }
  });
  return response.data.predictions;
}

export function createTrainingWebSocket(
  taskId: string,
  onMessage: (data: unknown) => void,
  onError?: (error: Event) => void
) {
  const ws = new WebSocket(`ws://localhost:8000/ws/train/${taskId}`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  ws.onerror = (error) => {
    onError?.(error);
  };

  return ws;
}
