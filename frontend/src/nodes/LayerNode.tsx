import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { LayerNodeData, LayerType } from '../types';
import './LayerNode.css';

const layerColors: Record<LayerType, string> = {
  input: '#52c41a',
  conv2d: '#1890ff',
  linear: '#722ed1',
  maxpool2d: '#13c2c2',
  batchnorm: '#fa8c16',
  dropout: '#eb2f96',
  flatten: '#faad14',
  relu: '#f5222d',
  sigmoid: '#a0d911',
  softmax: '#2f54eb',
  output: '#52c41a',
};

const layerIcons: Record<LayerType, string> = {
  input: '📥',
  conv2d: '🔲',
  linear: '➡️',
  maxpool2d: '⬇️',
  batchnorm: '📊',
  dropout: '🎲',
  flatten: '📏',
  relu: '⚡',
  sigmoid: '〰️',
  softmax: '📈',
  output: '📤',
};

type LayerNodeProps = NodeProps & {
  data: LayerNodeData;
};

function LayerNode({ data, selected }: LayerNodeProps) {
  const color = layerColors[data.layerType];
  const icon = layerIcons[data.layerType];

  const getParamsDisplay = () => {
    const params = data.params;
    switch (data.layerType) {
      case 'input':
        return `${params.channels}x${params.height}x${params.width}`;
      case 'conv2d':
        return `${params.out_channels} filters, ${params.kernel_size}x${params.kernel_size}`;
      case 'linear':
        return `${params.in_features} → ${params.out_features}`;
      case 'maxpool2d':
        return `${params.kernel_size}x${params.kernel_size}`;
      case 'batchnorm':
        return `${params.num_features} features`;
      case 'dropout':
        return `p=${params.p}`;
      case 'output':
        return `${params.num_classes} classes`;
      default:
        return '';
    }
  };

  return (
    <div
      className={`layer-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: color }}
    >
      {data.layerType !== 'input' && (
        <Handle type="target" position={Position.Top} />
      )}

      <div className="layer-node-header" style={{ backgroundColor: color }}>
        <span className="layer-icon">{icon}</span>
        <span className="layer-label">{data.label}</span>
      </div>

      <div className="layer-node-body">
        <span className="layer-params">{getParamsDisplay()}</span>
      </div>

      {data.layerType !== 'output' && (
        <Handle type="source" position={Position.Bottom} />
      )}
    </div>
  );
}

export default memo(LayerNode);
