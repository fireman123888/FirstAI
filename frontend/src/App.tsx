import { Layout, Button, Space, Tooltip } from 'antd';
import {
  ClearOutlined,
  GithubOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons';
import { useAppStore } from './store/useAppStore';
import Canvas from './components/Canvas/Canvas';
import NodePalette from './components/NodePalette/NodePalette';
import PropertyPanel from './components/PropertyPanel/PropertyPanel';
import CodePreview from './components/CodePreview/CodePreview';
import './App.css';

const { Header, Sider, Content } = Layout;

function App() {
  const { clearCanvas } = useAppStore();

  return (
    <Layout className="app-layout">
      <Header className="app-header">
        <div className="header-left">
          <h1 className="app-title">Neural Network Builder</h1>
          <span className="app-subtitle">可视化搭建神经网络</span>
        </div>
        <div className="header-right">
          <Space>
            <Tooltip title="清空画布">
              <Button icon={<ClearOutlined />} onClick={clearCanvas}>
                清空
              </Button>
            </Tooltip>
            <Tooltip title="帮助">
              <Button icon={<QuestionCircleOutlined />} />
            </Tooltip>
            <Tooltip title="GitHub">
              <Button icon={<GithubOutlined />} />
            </Tooltip>
          </Space>
        </div>
      </Header>

      <Layout className="app-body">
        <Sider width={280} className="left-sider">
          <NodePalette />
        </Sider>

        <Content className="main-content">
          <Canvas />
        </Content>

        <Sider width={320} className="right-sider">
          <div className="right-panels">
            <div className="property-section">
              <PropertyPanel />
            </div>
            <div className="code-section">
              <CodePreview />
            </div>
          </div>
        </Sider>
      </Layout>
    </Layout>
  );
}

export default App;
