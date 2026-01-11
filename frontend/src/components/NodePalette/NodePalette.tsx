import type { DragEvent } from 'react';
import { Card, Collapse, Button, Tooltip } from 'antd';
import {
  LoginOutlined,
  BorderOutlined,
  ArrowRightOutlined,
  VerticalAlignBottomOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  LogoutOutlined,
  AppstoreOutlined,
} from '@ant-design/icons';
import type { LayerType } from '../../types';
import { useAppStore } from '../../store/useAppStore';
import { modelTemplates } from '../../templates/modelTemplates';
import './NodePalette.css';

interface LayerItem {
  type: LayerType;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const inputOutputLayers: LayerItem[] = [
  {
    type: 'input',
    label: 'Input',
    icon: <LoginOutlined />,
    description: '输入层 - 定义输入数据形状',
  },
  {
    type: 'output',
    label: 'Output',
    icon: <LogoutOutlined />,
    description: '输出层 - 定义输出类别数',
  },
];

const coreLayers: LayerItem[] = [
  {
    type: 'conv2d',
    label: 'Conv2D',
    icon: <BorderOutlined />,
    description: '二维卷积层',
  },
  {
    type: 'linear',
    label: 'Linear',
    icon: <ArrowRightOutlined />,
    description: '全连接层',
  },
  {
    type: 'maxpool2d',
    label: 'MaxPool2D',
    icon: <VerticalAlignBottomOutlined />,
    description: '最大池化层',
  },
  {
    type: 'flatten',
    label: 'Flatten',
    icon: <ArrowRightOutlined />,
    description: '展平层',
  },
];

const normalizationLayers: LayerItem[] = [
  {
    type: 'batchnorm',
    label: 'BatchNorm',
    icon: <BarChartOutlined />,
    description: '批归一化层',
  },
  {
    type: 'dropout',
    label: 'Dropout',
    icon: <BarChartOutlined />,
    description: '随机失活层',
  },
];

const activationLayers: LayerItem[] = [
  {
    type: 'relu',
    label: 'ReLU',
    icon: <ThunderboltOutlined />,
    description: 'ReLU 激活函数',
  },
  {
    type: 'sigmoid',
    label: 'Sigmoid',
    icon: <ThunderboltOutlined />,
    description: 'Sigmoid 激活函数',
  },
  {
    type: 'softmax',
    label: 'Softmax',
    icon: <ThunderboltOutlined />,
    description: 'Softmax 激活函数',
  },
];

interface LayerCardProps {
  layer: LayerItem;
}

function LayerCard({ layer }: LayerCardProps) {
  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/reactflow', layer.type);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      className="layer-card"
      draggable
      onDragStart={onDragStart}
    >
      <div className="layer-card-icon">{layer.icon}</div>
      <div className="layer-card-content">
        <div className="layer-card-label">{layer.label}</div>
        <div className="layer-card-desc">{layer.description}</div>
      </div>
    </div>
  );
}

export default function NodePalette() {
  const { setNodes, setEdges } = useAppStore();

  const loadTemplate = (templateIndex: number) => {
    const template = modelTemplates[templateIndex];
    setNodes(template.nodes);
    setEdges(template.edges);
  };

  const items = [
    {
      key: 'templates',
      label: (
        <span>
          <AppstoreOutlined style={{ marginRight: 8 }} />
          模型模板
        </span>
      ),
      children: (
        <div className="template-list">
          {modelTemplates.map((template, index) => (
            <Tooltip key={template.name} title={template.description} placement="right">
              <Button
                block
                onClick={() => loadTemplate(index)}
                className="template-btn"
              >
                {template.name}
              </Button>
            </Tooltip>
          ))}
        </div>
      ),
    },
    {
      key: 'io',
      label: '输入/输出',
      children: (
        <div className="layer-list">
          {inputOutputLayers.map((layer) => (
            <LayerCard key={layer.type} layer={layer} />
          ))}
        </div>
      ),
    },
    {
      key: 'core',
      label: '核心层',
      children: (
        <div className="layer-list">
          {coreLayers.map((layer) => (
            <LayerCard key={layer.type} layer={layer} />
          ))}
        </div>
      ),
    },
    {
      key: 'normalization',
      label: '正则化层',
      children: (
        <div className="layer-list">
          {normalizationLayers.map((layer) => (
            <LayerCard key={layer.type} layer={layer} />
          ))}
        </div>
      ),
    },
    {
      key: 'activation',
      label: '激活函数',
      children: (
        <div className="layer-list">
          {activationLayers.map((layer) => (
            <LayerCard key={layer.type} layer={layer} />
          ))}
        </div>
      ),
    },
  ];

  return (
    <Card title="网络层组件" className="node-palette" size="small">
      <p className="drag-hint">选择模板或拖拽组件到画布</p>
      <Collapse items={items} defaultActiveKey={['templates', 'io', 'core']} ghost />
    </Card>
  );
}
