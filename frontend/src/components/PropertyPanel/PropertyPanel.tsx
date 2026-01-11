import { Card, Form, InputNumber, Empty, Slider, Divider, Button } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { useAppStore } from '../../store/useAppStore';
import type { LayerType, LayerParams } from '../../types';
import './PropertyPanel.css';

export default function PropertyPanel() {
  const { selectedNode, updateNodeData, nodes, setNodes, edges, setEdges } = useAppStore();

  if (!selectedNode) {
    return (
      <Card title="属性面板" className="property-panel" size="small">
        <Empty description="选择一个节点查看属性" />
      </Card>
    );
  }

  const { layerType, params } = selectedNode.data;

  const handleParamChange = (key: string, value: number) => {
    updateNodeData(selectedNode.id, {
      params: { ...params, [key]: value } as LayerParams,
    });
  };

  const handleDeleteNode = () => {
    setNodes(nodes.filter((n) => n.id !== selectedNode.id));
    setEdges(edges.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
  };

  const renderParamsForm = () => {
    switch (layerType) {
      case 'input':
        return (
          <>
            <Form.Item label="通道数">
              <InputNumber
                min={1}
                max={512}
                value={params.channels as number}
                onChange={(v) => handleParamChange('channels', v || 1)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="高度">
              <InputNumber
                min={1}
                max={1024}
                value={params.height as number}
                onChange={(v) => handleParamChange('height', v || 28)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="宽度">
              <InputNumber
                min={1}
                max={1024}
                value={params.width as number}
                onChange={(v) => handleParamChange('width', v || 28)}
                style={{ width: '100%' }}
              />
            </Form.Item>
          </>
        );

      case 'conv2d':
        return (
          <>
            <Form.Item label="输入通道">
              <InputNumber
                min={1}
                max={512}
                value={params.in_channels as number}
                onChange={(v) => handleParamChange('in_channels', v || 1)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="输出通道">
              <InputNumber
                min={1}
                max={512}
                value={params.out_channels as number}
                onChange={(v) => handleParamChange('out_channels', v || 32)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="卷积核大小">
              <InputNumber
                min={1}
                max={15}
                value={params.kernel_size as number}
                onChange={(v) => handleParamChange('kernel_size', v || 3)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="步长">
              <InputNumber
                min={1}
                max={5}
                value={params.stride as number}
                onChange={(v) => handleParamChange('stride', v || 1)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="填充">
              <InputNumber
                min={0}
                max={10}
                value={params.padding as number}
                onChange={(v) => handleParamChange('padding', v ?? 0)}
                style={{ width: '100%' }}
              />
            </Form.Item>
          </>
        );

      case 'linear':
        return (
          <>
            <Form.Item label="输入特征">
              <InputNumber
                min={1}
                max={65536}
                value={params.in_features as number}
                onChange={(v) => handleParamChange('in_features', v || 128)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="输出特征">
              <InputNumber
                min={1}
                max={65536}
                value={params.out_features as number}
                onChange={(v) => handleParamChange('out_features', v || 64)}
                style={{ width: '100%' }}
              />
            </Form.Item>
          </>
        );

      case 'maxpool2d':
        return (
          <>
            <Form.Item label="池化核大小">
              <InputNumber
                min={1}
                max={10}
                value={params.kernel_size as number}
                onChange={(v) => handleParamChange('kernel_size', v || 2)}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="步长">
              <InputNumber
                min={1}
                max={10}
                value={params.stride as number}
                onChange={(v) => handleParamChange('stride', v || 2)}
                style={{ width: '100%' }}
              />
            </Form.Item>
          </>
        );

      case 'batchnorm':
        return (
          <Form.Item label="特征数">
            <InputNumber
              min={1}
              max={512}
              value={params.num_features as number}
              onChange={(v) => handleParamChange('num_features', v || 32)}
              style={{ width: '100%' }}
            />
          </Form.Item>
        );

      case 'dropout':
        return (
          <Form.Item label="丢弃率">
            <Slider
              min={0}
              max={1}
              step={0.1}
              value={params.p as number}
              onChange={(v) => handleParamChange('p', v)}
            />
          </Form.Item>
        );

      case 'output':
        return (
          <Form.Item label="类别数">
            <InputNumber
              min={1}
              max={10000}
              value={params.num_classes as number}
              onChange={(v) => handleParamChange('num_classes', v || 10)}
              style={{ width: '100%' }}
            />
          </Form.Item>
        );

      default:
        return <p className="no-params">该层无可配置参数</p>;
    }
  };

  const layerLabels: Record<LayerType, string> = {
    input: 'Input 输入层',
    conv2d: 'Conv2D 卷积层',
    linear: 'Linear 全连接层',
    maxpool2d: 'MaxPool2D 池化层',
    batchnorm: 'BatchNorm 批归一化',
    dropout: 'Dropout 随机失活',
    flatten: 'Flatten 展平层',
    relu: 'ReLU 激活函数',
    sigmoid: 'Sigmoid 激活函数',
    softmax: 'Softmax 激活函数',
    output: 'Output 输出层',
  };

  return (
    <Card
      title="属性面板"
      className="property-panel"
      size="small"
      extra={
        <Button
          type="text"
          danger
          icon={<DeleteOutlined />}
          onClick={handleDeleteNode}
          size="small"
        >
          删除
        </Button>
      }
    >
      <div className="layer-info">
        <span className="layer-type-label">{layerLabels[layerType]}</span>
        <span className="layer-id">ID: {selectedNode.id}</span>
      </div>
      <Divider style={{ margin: '12px 0' }} />
      <Form layout="vertical" size="small">
        {renderParamsForm()}
      </Form>
    </Card>
  );
}
