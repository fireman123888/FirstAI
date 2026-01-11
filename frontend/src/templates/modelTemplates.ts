import type { Node, Edge } from '@xyflow/react';
import type { LayerNodeData } from '../types';

export interface ModelTemplate {
  name: string;
  description: string;
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
}

// LeNet-5 模板
export const leNetTemplate: ModelTemplate = {
  name: 'LeNet-5',
  description: '经典手写数字识别网络',
  nodes: [
    {
      id: 'input_0',
      type: 'layerNode',
      position: { x: 250, y: 0 },
      data: { label: 'Input', layerType: 'input', params: { channels: 1, height: 28, width: 28 } },
    },
    {
      id: 'conv_1',
      type: 'layerNode',
      position: { x: 250, y: 100 },
      data: { label: 'Conv2D', layerType: 'conv2d', params: { in_channels: 1, out_channels: 6, kernel_size: 5, stride: 1, padding: 2 } },
    },
    {
      id: 'relu_1',
      type: 'layerNode',
      position: { x: 250, y: 200 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'pool_1',
      type: 'layerNode',
      position: { x: 250, y: 300 },
      data: { label: 'MaxPool2D', layerType: 'maxpool2d', params: { kernel_size: 2, stride: 2 } },
    },
    {
      id: 'conv_2',
      type: 'layerNode',
      position: { x: 250, y: 400 },
      data: { label: 'Conv2D', layerType: 'conv2d', params: { in_channels: 6, out_channels: 16, kernel_size: 5, stride: 1, padding: 0 } },
    },
    {
      id: 'relu_2',
      type: 'layerNode',
      position: { x: 250, y: 500 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'pool_2',
      type: 'layerNode',
      position: { x: 250, y: 600 },
      data: { label: 'MaxPool2D', layerType: 'maxpool2d', params: { kernel_size: 2, stride: 2 } },
    },
    {
      id: 'flatten',
      type: 'layerNode',
      position: { x: 250, y: 700 },
      data: { label: 'Flatten', layerType: 'flatten', params: {} },
    },
    {
      id: 'fc_1',
      type: 'layerNode',
      position: { x: 250, y: 800 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 400, out_features: 120 } },
    },
    {
      id: 'relu_3',
      type: 'layerNode',
      position: { x: 250, y: 900 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'fc_2',
      type: 'layerNode',
      position: { x: 250, y: 1000 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 120, out_features: 84 } },
    },
    {
      id: 'relu_4',
      type: 'layerNode',
      position: { x: 250, y: 1100 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'fc_3',
      type: 'layerNode',
      position: { x: 250, y: 1200 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 84, out_features: 10 } },
    },
    {
      id: 'output',
      type: 'layerNode',
      position: { x: 250, y: 1300 },
      data: { label: 'Output', layerType: 'output', params: { num_classes: 10 } },
    },
  ],
  edges: [
    { id: 'e0-1', source: 'input_0', target: 'conv_1', animated: true },
    { id: 'e1-2', source: 'conv_1', target: 'relu_1', animated: true },
    { id: 'e2-3', source: 'relu_1', target: 'pool_1', animated: true },
    { id: 'e3-4', source: 'pool_1', target: 'conv_2', animated: true },
    { id: 'e4-5', source: 'conv_2', target: 'relu_2', animated: true },
    { id: 'e5-6', source: 'relu_2', target: 'pool_2', animated: true },
    { id: 'e6-7', source: 'pool_2', target: 'flatten', animated: true },
    { id: 'e7-8', source: 'flatten', target: 'fc_1', animated: true },
    { id: 'e8-9', source: 'fc_1', target: 'relu_3', animated: true },
    { id: 'e9-10', source: 'relu_3', target: 'fc_2', animated: true },
    { id: 'e10-11', source: 'fc_2', target: 'relu_4', animated: true },
    { id: 'e11-12', source: 'relu_4', target: 'fc_3', animated: true },
    { id: 'e12-13', source: 'fc_3', target: 'output', animated: true },
  ],
};

// 简单 MLP 模板
export const simpleMlpTemplate: ModelTemplate = {
  name: '简单 MLP',
  description: '三层全连接网络',
  nodes: [
    {
      id: 'input_0',
      type: 'layerNode',
      position: { x: 250, y: 0 },
      data: { label: 'Input', layerType: 'input', params: { channels: 1, height: 28, width: 28 } },
    },
    {
      id: 'flatten',
      type: 'layerNode',
      position: { x: 250, y: 100 },
      data: { label: 'Flatten', layerType: 'flatten', params: {} },
    },
    {
      id: 'fc_1',
      type: 'layerNode',
      position: { x: 250, y: 200 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 784, out_features: 256 } },
    },
    {
      id: 'relu_1',
      type: 'layerNode',
      position: { x: 250, y: 300 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'dropout_1',
      type: 'layerNode',
      position: { x: 250, y: 400 },
      data: { label: 'Dropout', layerType: 'dropout', params: { p: 0.5 } },
    },
    {
      id: 'fc_2',
      type: 'layerNode',
      position: { x: 250, y: 500 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 256, out_features: 128 } },
    },
    {
      id: 'relu_2',
      type: 'layerNode',
      position: { x: 250, y: 600 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'fc_3',
      type: 'layerNode',
      position: { x: 250, y: 700 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 128, out_features: 10 } },
    },
    {
      id: 'output',
      type: 'layerNode',
      position: { x: 250, y: 800 },
      data: { label: 'Output', layerType: 'output', params: { num_classes: 10 } },
    },
  ],
  edges: [
    { id: 'e0-1', source: 'input_0', target: 'flatten', animated: true },
    { id: 'e1-2', source: 'flatten', target: 'fc_1', animated: true },
    { id: 'e2-3', source: 'fc_1', target: 'relu_1', animated: true },
    { id: 'e3-4', source: 'relu_1', target: 'dropout_1', animated: true },
    { id: 'e4-5', source: 'dropout_1', target: 'fc_2', animated: true },
    { id: 'e5-6', source: 'fc_2', target: 'relu_2', animated: true },
    { id: 'e6-7', source: 'relu_2', target: 'fc_3', animated: true },
    { id: 'e7-8', source: 'fc_3', target: 'output', animated: true },
  ],
};

// 简单 CNN 模板
export const simpleCnnTemplate: ModelTemplate = {
  name: '简单 CNN',
  description: '两层卷积网络',
  nodes: [
    {
      id: 'input_0',
      type: 'layerNode',
      position: { x: 250, y: 0 },
      data: { label: 'Input', layerType: 'input', params: { channels: 1, height: 28, width: 28 } },
    },
    {
      id: 'conv_1',
      type: 'layerNode',
      position: { x: 250, y: 100 },
      data: { label: 'Conv2D', layerType: 'conv2d', params: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } },
    },
    {
      id: 'relu_1',
      type: 'layerNode',
      position: { x: 250, y: 200 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'pool_1',
      type: 'layerNode',
      position: { x: 250, y: 300 },
      data: { label: 'MaxPool2D', layerType: 'maxpool2d', params: { kernel_size: 2, stride: 2 } },
    },
    {
      id: 'conv_2',
      type: 'layerNode',
      position: { x: 250, y: 400 },
      data: { label: 'Conv2D', layerType: 'conv2d', params: { in_channels: 32, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 } },
    },
    {
      id: 'relu_2',
      type: 'layerNode',
      position: { x: 250, y: 500 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'pool_2',
      type: 'layerNode',
      position: { x: 250, y: 600 },
      data: { label: 'MaxPool2D', layerType: 'maxpool2d', params: { kernel_size: 2, stride: 2 } },
    },
    {
      id: 'flatten',
      type: 'layerNode',
      position: { x: 250, y: 700 },
      data: { label: 'Flatten', layerType: 'flatten', params: {} },
    },
    {
      id: 'fc_1',
      type: 'layerNode',
      position: { x: 250, y: 800 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 3136, out_features: 128 } },
    },
    {
      id: 'relu_3',
      type: 'layerNode',
      position: { x: 250, y: 900 },
      data: { label: 'ReLU', layerType: 'relu', params: {} },
    },
    {
      id: 'fc_2',
      type: 'layerNode',
      position: { x: 250, y: 1000 },
      data: { label: 'Linear', layerType: 'linear', params: { in_features: 128, out_features: 10 } },
    },
    {
      id: 'output',
      type: 'layerNode',
      position: { x: 250, y: 1100 },
      data: { label: 'Output', layerType: 'output', params: { num_classes: 10 } },
    },
  ],
  edges: [
    { id: 'e0-1', source: 'input_0', target: 'conv_1', animated: true },
    { id: 'e1-2', source: 'conv_1', target: 'relu_1', animated: true },
    { id: 'e2-3', source: 'relu_1', target: 'pool_1', animated: true },
    { id: 'e3-4', source: 'pool_1', target: 'conv_2', animated: true },
    { id: 'e4-5', source: 'conv_2', target: 'relu_2', animated: true },
    { id: 'e5-6', source: 'relu_2', target: 'pool_2', animated: true },
    { id: 'e6-7', source: 'pool_2', target: 'flatten', animated: true },
    { id: 'e7-8', source: 'flatten', target: 'fc_1', animated: true },
    { id: 'e8-9', source: 'fc_1', target: 'relu_3', animated: true },
    { id: 'e9-10', source: 'relu_3', target: 'fc_2', animated: true },
    { id: 'e10-11', source: 'fc_2', target: 'output', animated: true },
  ],
};

export const modelTemplates: ModelTemplate[] = [
  simpleMlpTemplate,
  simpleCnnTemplate,
  leNetTemplate,
];
