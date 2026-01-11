import { useState, useEffect } from 'react';
import { Card, Button, Space, message, Spin, Tabs, InputNumber, Form, Select, Progress } from 'antd';
import {
  CodeOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  CopyOutlined,
  SettingOutlined,
  LineChartOutlined,
  TrophyOutlined,
  StarFilled,
  CheckCircleFilled,
  BulbOutlined,
  ReloadOutlined,
  CheckOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useAppStore } from '../../store/useAppStore';
import { generateCode, startTraining, getPredictions, API_BASE, WS_BASE } from '../../api';
import type { PredictionResult } from '../../api';
import SyntaxHighlighter from './SyntaxHighlighter';
import './CodePreview.css';

export default function CodePreview() {
  const {
    nodes,
    edges,
    generatedCode,
    setGeneratedCode,
    trainingConfig,
    setTrainingConfig,
    trainingStatus,
    setTrainingStatus,
  } = useAppStore();

  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('code');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loadingPredictions, setLoadingPredictions] = useState(false);

  // 训练完成时自动切换到结果页并获取预测示例
  useEffect(() => {
    if (trainingStatus.status === 'completed' && taskId) {
      setActiveTab('result');
      fetchPredictions();
    }
  }, [trainingStatus.status, taskId]);

  const fetchPredictions = async () => {
    if (!taskId) return;
    setLoadingPredictions(true);
    try {
      const results = await getPredictions(taskId, 6);
      setPredictions(results);
    } catch (error) {
      console.error('获取预测示例失败:', error);
    } finally {
      setLoadingPredictions(false);
    }
  };

  const handleGenerate = async () => {
    if (nodes.length === 0) {
      message.warning('请先在画布上添加网络层');
      return;
    }

    setLoading(true);
    try {
      const code = await generateCode(nodes, edges);
      setGeneratedCode(code);
      message.success('代码生成成功');
    } catch (error: unknown) {
      const err = error as Error;
      message.error(err.message || '生成失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (generatedCode) {
      navigator.clipboard.writeText(generatedCode);
      message.success('已复制到剪贴板');
    }
  };

  const handleDownload = () => {
    if (generatedCode) {
      const blob = new Blob([generatedCode], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model.py';
      a.click();
      URL.revokeObjectURL(url);
      message.success('下载成功');
    }
  };

  const handleTrain = async () => {
    if (!generatedCode) {
      message.warning('请先生成代码');
      return;
    }

    if (nodes.length === 0) {
      message.warning('请先添加网络层');
      return;
    }

    try {
      // 清除之前的预测结果
      setPredictions([]);

      // 设置初始状态
      setTrainingStatus({
        status: 'running',
        current_epoch: 0,
        total_epochs: trainingConfig.epochs,
        loss: 0,
        accuracy: 0,
        history: [],
      });

      const { task_id } = await startTraining(nodes, edges, trainingConfig);
      setTaskId(task_id);
      message.success('训练任务已启动');

      // 连接 WebSocket 获取实时更新
      const ws = new WebSocket(`${WS_BASE}/ws/train/${task_id}`);

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) {
          message.error(data.error);
          setTrainingStatus({ status: 'failed' });
          ws.close();
          return;
        }

        setTrainingStatus({
          status: data.status,
          current_epoch: data.current_epoch,
          total_epochs: data.total_epochs,
          loss: data.loss,
          accuracy: data.accuracy,
          history: data.history || [],
        });

        if (data.status === 'completed') {
          message.success('训练完成！');
          ws.close();
        } else if (data.status === 'failed') {
          message.error('训练失败：' + (data.error || '未知错误'));
          ws.close();
        }
      };

      ws.onerror = () => {
        // WebSocket 失败时使用轮询
        const pollStatus = async () => {
          try {
            const response = await fetch(`${API_BASE}/train/${task_id}`);
            const data = await response.json();

            setTrainingStatus({
              status: data.status,
              current_epoch: data.current_epoch,
              total_epochs: data.total_epochs,
              loss: data.loss,
              accuracy: data.accuracy,
              history: data.history || [],
            });

            if (data.status === 'running') {
              setTimeout(pollStatus, 1000);
            } else if (data.status === 'completed') {
              message.success('训练完成！');
            } else if (data.status === 'failed') {
              message.error('训练失败');
            }
          } catch {
            setTimeout(pollStatus, 2000);
          }
        };
        pollStatus();
      };

      // 切换到训练曲线标签
      setActiveTab('chart');

    } catch (error: unknown) {
      const err = error as Error;
      message.error(err.message || '启动训练失败');
      setTrainingStatus({ status: 'idle' });
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return '训练中';
      case 'completed': return '已完成';
      case 'failed': return '失败';
      default: return '空闲';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#1890ff';
      case 'completed': return '#52c41a';
      case 'failed': return '#ff4d4f';
      default: return '#999';
    }
  };

  // 获取模型评级
  const getPerformanceRating = (accuracy: number) => {
    if (accuracy >= 0.98) return { level: 'SSS', color: '#ff4500', text: '超神！', stars: 5 };
    if (accuracy >= 0.95) return { level: 'S', color: '#ff6b00', text: '卓越', stars: 5 };
    if (accuracy >= 0.90) return { level: 'A', color: '#52c41a', text: '优秀', stars: 4 };
    if (accuracy >= 0.80) return { level: 'B', color: '#1890ff', text: '良好', stars: 3 };
    if (accuracy >= 0.70) return { level: 'C', color: '#faad14', text: '及格', stars: 2 };
    return { level: 'D', color: '#ff4d4f', text: '需要改进', stars: 1 };
  };

  // 获取通俗解释
  const getSimpleExplanation = (accuracy: number) => {
    const correct = Math.round(accuracy * 100);
    if (accuracy >= 0.95) {
      return `你的模型超级厉害！给它看100张手写数字图片，它能认对${correct}张！这已经接近人类水平了！`;
    }
    if (accuracy >= 0.90) {
      return `你的模型很棒！给它看100张手写数字图片，它能认对${correct}张。继续优化可以做得更好！`;
    }
    if (accuracy >= 0.80) {
      return `你的模型不错！给它看100张手写数字图片，它能认对${correct}张。试试增加训练轮数或调整网络结构？`;
    }
    if (accuracy >= 0.70) {
      return `你的模型及格了！给它看100张手写数字图片，它能认对${correct}张。建议增加更多的网络层来提升效果。`;
    }
    return `你的模型还在学习中，给它看100张手写数字图片，只能认对${correct}张。试试使用模板中的 LeNet 结构？`;
  };

  // 获取改进建议
  const getImprovementTips = (accuracy: number) => {
    if (accuracy >= 0.95) {
      return [
        '你已经做得很好了！',
        '可以尝试用更少的参数达到同样效果',
        '或者挑战更难的数据集',
      ];
    }
    if (accuracy >= 0.85) {
      return [
        '增加卷积层可以提取更多特征',
        '试试增加训练轮数到20-30轮',
        '可以添加 BatchNorm 层加速训练',
      ];
    }
    return [
      '建议使用 Conv2D + MaxPool 的经典组合',
      '确保网络包含足够的层数（至少3-4层）',
      '试试使用预设的 LeNet 模板',
      '增加训练轮数让模型学习更充分',
    ];
  };

  // 渲染星星
  const renderStars = (count: number) => {
    return Array(5).fill(0).map((_, i) => (
      <StarFilled
        key={i}
        style={{
          color: i < count ? '#fadb14' : '#d9d9d9',
          fontSize: 20,
          marginRight: 2
        }}
      />
    ));
  };

  const rating = getPerformanceRating(trainingStatus.accuracy);

  const tabItems = [
    {
      key: 'code',
      label: (
        <span>
          <CodeOutlined />
          代码预览
        </span>
      ),
      children: (
        <div className="code-container">
          {loading ? (
            <div className="loading-container">
              <Spin tip="AI 正在生成代码..." />
            </div>
          ) : generatedCode ? (
            <SyntaxHighlighter code={generatedCode} language="python" />
          ) : (
            <div className="empty-code">
              <CodeOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
              <p>点击"生成代码"按钮生成 PyTorch 代码</p>
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'training',
      label: (
        <span>
          <SettingOutlined />
          训练配置
        </span>
      ),
      children: (
        <div className="training-config">
          <Form layout="vertical" size="small">
            <Form.Item label="训练轮数 (Epochs)">
              <InputNumber
                min={1}
                max={100}
                value={trainingConfig.epochs}
                onChange={(v) => setTrainingConfig({ epochs: v || 10 })}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="学习率 (Learning Rate)">
              <InputNumber
                min={0.00001}
                max={1}
                step={0.001}
                value={trainingConfig.learning_rate}
                onChange={(v) => setTrainingConfig({ learning_rate: v || 0.001 })}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="批大小 (Batch Size)">
              <InputNumber
                min={1}
                max={512}
                value={trainingConfig.batch_size}
                onChange={(v) => setTrainingConfig({ batch_size: v || 64 })}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="优化器">
              <Select
                value={trainingConfig.optimizer}
                onChange={(v) => setTrainingConfig({ optimizer: v })}
              >
                <Select.Option value="adam">Adam</Select.Option>
                <Select.Option value="sgd">SGD</Select.Option>
              </Select>
            </Form.Item>
          </Form>

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleTrain}
            loading={trainingStatus.status === 'running'}
            block
          >
            {trainingStatus.status === 'running' ? '训练中...' : '开始训练'}
          </Button>
        </div>
      ),
    },
    {
      key: 'chart',
      label: (
        <span>
          <LineChartOutlined />
          训练曲线
        </span>
      ),
      children: (
        <div className="training-chart">
          {trainingStatus.status !== 'idle' && (
            <>
              <div className="training-progress">
                <div className="progress-header">
                  <span style={{ color: getStatusColor(trainingStatus.status) }}>
                    {getStatusText(trainingStatus.status)}
                  </span>
                  <span>
                    Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs}
                  </span>
                </div>
                <Progress
                  percent={Math.round((trainingStatus.current_epoch / trainingStatus.total_epochs) * 100)}
                  status={trainingStatus.status === 'running' ? 'active' : 'normal'}
                  strokeColor={getStatusColor(trainingStatus.status)}
                />
                <div className="metrics-row">
                  <div className="metric-item">
                    <span className="metric-label">Loss</span>
                    <span className="metric-value">{trainingStatus.loss.toFixed(4)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">{(trainingStatus.accuracy * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>

              {trainingStatus.history.length > 0 && (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={trainingStatus.history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'Epoch', position: 'bottom', fontSize: 10 }}
                      />
                      <YAxis
                        yAxisId="left"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 10 }}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: 10 }}
                        domain={[0, 1]}
                        label={{ value: 'Accuracy', angle: 90, position: 'insideRight', fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{ fontSize: 12 }}
                        formatter={(value, name) => {
                          if (typeof value !== 'number') return [String(value), name];
                          return [
                            name === 'accuracy' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4),
                            name === 'accuracy' ? 'Accuracy' : 'Loss'
                          ];
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="loss"
                        stroke="#ff7875"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="Loss"
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#52c41a"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name="Accuracy"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}

          {trainingStatus.status === 'idle' && (
            <div className="empty-chart">
              <LineChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
              <p>开始训练后显示曲线</p>
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'result',
      label: (
        <span>
          <TrophyOutlined />
          训练结果
        </span>
      ),
      children: (
        <div className="training-result">
          {trainingStatus.status === 'completed' ? (
            <div className="result-content">
              {/* 评级展示 */}
              <div className="rating-section">
                <div
                  className="rating-badge"
                  style={{
                    background: `linear-gradient(135deg, ${rating.color}, ${rating.color}88)`,
                  }}
                >
                  <span className="rating-level">{rating.level}</span>
                  <span className="rating-text">{rating.text}</span>
                </div>
                <div className="stars-row">
                  {renderStars(rating.stars)}
                </div>
              </div>

              {/* 准确率大字展示 */}
              <div className="accuracy-display">
                <div className="accuracy-number">
                  {(trainingStatus.accuracy * 100).toFixed(1)}
                  <span className="accuracy-percent">%</span>
                </div>
                <div className="accuracy-label">识别准确率</div>
              </div>

              {/* 预测示例展示 */}
              <div className="predictions-section">
                <div className="section-title">
                  <CheckCircleFilled style={{ color: '#1890ff', marginRight: 8 }} />
                  模型识别示例
                  <Button
                    type="link"
                    size="small"
                    icon={<ReloadOutlined />}
                    onClick={fetchPredictions}
                    loading={loadingPredictions}
                    style={{ marginLeft: 'auto' }}
                  >
                    换一批
                  </Button>
                </div>
                {loadingPredictions ? (
                  <div className="predictions-loading">
                    <Spin tip="加载中..." />
                  </div>
                ) : predictions.length > 0 ? (
                  <div className="predictions-grid">
                    {predictions.map((pred, index) => (
                      <div
                        key={index}
                        className={`prediction-card ${pred.correct ? 'correct' : 'wrong'}`}
                      >
                        <img
                          src={`data:image/png;base64,${pred.image}`}
                          alt={`数字 ${pred.label}`}
                          className="prediction-image"
                        />
                        <div className="prediction-info">
                          <div className="prediction-row">
                            <span className="info-label">实际:</span>
                            <span className="info-value">{pred.label}</span>
                          </div>
                          <div className="prediction-row">
                            <span className="info-label">预测:</span>
                            <span className="info-value">{pred.prediction}</span>
                            {pred.correct ? (
                              <CheckOutlined style={{ color: '#52c41a', marginLeft: 4 }} />
                            ) : (
                              <CloseOutlined style={{ color: '#ff4d4f', marginLeft: 4 }} />
                            )}
                          </div>
                          <div className="prediction-confidence">
                            置信度: {pred.confidence}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="predictions-empty">点击"换一批"查看预测示例</div>
                )}
              </div>

              {/* 通俗解释 */}
              <div className="explanation-section">
                <div className="section-title">
                  <BulbOutlined style={{ color: '#faad14', marginRight: 8 }} />
                  这意味着什么？
                </div>
                <p className="explanation-text">
                  {getSimpleExplanation(trainingStatus.accuracy)}
                </p>
              </div>

              {/* 训练统计 */}
              <div className="stats-section">
                <div className="stat-item">
                  <span className="stat-label">训练轮数</span>
                  <span className="stat-value">{trainingStatus.total_epochs} 轮</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">最终损失</span>
                  <span className="stat-value">{trainingStatus.loss.toFixed(4)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">网络层数</span>
                  <span className="stat-value">{nodes.length} 层</span>
                </div>
              </div>

              {/* 改进建议 */}
              <div className="tips-section">
                <div className="section-title">
                  <BulbOutlined style={{ color: '#faad14', marginRight: 8 }} />
                  小贴士
                </div>
                <ul className="tips-list">
                  {getImprovementTips(trainingStatus.accuracy).map((tip, index) => (
                    <li key={index}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : trainingStatus.status === 'failed' ? (
            <div className="empty-result error">
              <TrophyOutlined style={{ fontSize: 48, color: '#ff4d4f' }} />
              <p>训练失败了，请检查网络结构后重试</p>
            </div>
          ) : trainingStatus.status === 'running' ? (
            <div className="empty-result">
              <Spin size="large" />
              <p>训练进行中，请稍候...</p>
            </div>
          ) : (
            <div className="empty-result">
              <TrophyOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
              <p>完成训练后这里会显示详细结果</p>
            </div>
          )}
        </div>
      ),
    },
  ];

  return (
    <Card
      title="代码 & 训练"
      className="code-preview"
      size="small"
      extra={
        <Space>
          <Button
            type="primary"
            icon={<CodeOutlined />}
            onClick={handleGenerate}
            loading={loading}
            size="small"
          >
            生成代码
          </Button>
        </Space>
      }
    >
      <div className="action-buttons">
        <Button
          icon={<CopyOutlined />}
          onClick={handleCopy}
          disabled={!generatedCode}
          size="small"
        >
          复制
        </Button>
        <Button
          icon={<DownloadOutlined />}
          onClick={handleDownload}
          disabled={!generatedCode}
          size="small"
        >
          下载
        </Button>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        size="small"
      />
    </Card>
  );
}
