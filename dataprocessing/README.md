# 广告费用与产品销量（回归建模）

说明
- 这是一个基于线性回归与正则化回归（岭回归、Lasso、弹性网络）的示例脚本，旨在演示如何用广告投放数据预测产品销量并比较不同模型的性能。
- 我尽量保留了你原始代码的结构、可视化和评估逻辑；新增内容仅用于保证在没有原始数据时也能运行（会生成合成数据并保存为 `data/Advertising.csv`）。

主要文件
- `1 广告费用与产品销量.py`：主脚本，包含数据加载、预处理、模型训练、调参和可视化。
- `data/Advertising.csv`：如果运行时未检测到原始数据，脚本会自动生成并写入此文件。
- `requirements.txt`：运行所需的 Python 包列表。

快速开始
1. 安装依赖（推荐使用虚拟环境）：

```bash
pip install -r requirements.txt
```

2. 运行脚本：

```bash
python "1 广告费用与产品销量.py"
```

如果你想生成（或重新生成）更真实的合成数据：

```bash
python generate_ad_data.py  # 生成默认 100 条样本
# 或者指定样本数和随机种子：
python generate_ad_data.py 200 42
```

运行结果
- 脚本会在当前目录下保存三张图片：
  - `model_performance_comparison.png`（模型性能比较）
  - `prediction_comparison.png`（预测值与真实值对比）
  - `feature_importance.png`（线性模型系数/特征重要性）
- 脚本会在控制台打印各模型的评估指标、最佳模型及其系数解释建议。

数据说明（合成数据）
- 若脚本找不到 `data/Advertising.csv`，会生成 N=200 的合成样本，特征为 `TV`, `radio`, `newspaper`，目标为 `sales`。
- 合成规则为线性组合加噪声，便于复现回归行为并保留原始分析流程。

设计原则与保留
- 保留了原代码的模型列表、网格搜索、评估指标及可视化模块，便于你继续改进或替换数据源。
- 对数据列名做了容错（支持 `sales` 或 `Sales`）。

如需我将该脚本封装为可运行的仓库（包含 `.gitignore`、更详细的说明或CI），告诉我你想要的目标，我可以继续完善。