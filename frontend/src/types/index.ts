// 神经网络层类型定义

export type LayerType =
  | 'input'
  | 'conv2d'
  | 'linear'
  | 'maxpool2d'
  | 'batchnorm'
  | 'dropout'
  | 'flatten'
  | 'relu'
  | 'sigmoid'
  | 'softmax'
  | 'output';

export interface InputParams {
  channels: number;
  height: number;
  width: number;
  [key: string]: unknown;
}

export interface Conv2dParams {
  in_channels: number;
  out_channels: number;
  kernel_size: number;
  stride: number;
  padding: number;
  [key: string]: unknown;
}

export interface LinearParams {
  in_features: number;
  out_features: number;
  [key: string]: unknown;
}

export interface MaxPool2dParams {
  kernel_size: number;
  stride: number;
  [key: string]: unknown;
}

export interface BatchNormParams {
  num_features: number;
  [key: string]: unknown;
}

export interface DropoutParams {
  p: number;
  [key: string]: unknown;
}

export interface OutputParams {
  num_classes: number;
  [key: string]: unknown;
}

export type LayerParams =
  | InputParams
  | Conv2dParams
  | LinearParams
  | MaxPool2dParams
  | BatchNormParams
  | DropoutParams
  | OutputParams
  | Record<string, unknown>;

export interface LayerNodeData extends Record<string, unknown> {
  label: string;
  layerType: LayerType;
  params: LayerParams;
}

export interface TrainingConfig {
  epochs: number;
  learning_rate: number;
  batch_size: number;
  optimizer: 'adam' | 'sgd';
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  current_epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  history: { epoch: number; loss: number; accuracy: number }[];
}
