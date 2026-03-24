import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 加载数据
def load_data(file_path):
    """加载数据集"""
    try:
        df = pd.read_csv(file_path)
        print("数据基本信息：")
        # 强制将关键列转换为数值，遇错值设为 NaN，然后删除包含 NaN 的行
        for col in ['TV', 'radio', 'newspaper', 'sales']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        before = len(df)
        df = df.dropna(subset=[c for c in ['TV', 'radio', 'newspaper', 'sales'] if c in df.columns])
        after = len(df)
        if before != after:
            print(f"已删除 {before-after} 行包含无效数值的记录")
        df.info()
        return df
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None


def generate_synthetic_data(file_path, n_samples=200, random_state=42):
    """当找不到原始数据时，生成合成Advertising数据并保存为CSV。"""
    rng = np.random.RandomState(random_state)
    TV = rng.uniform(0, 300, size=n_samples)
    radio = rng.uniform(0, 50, size=n_samples)
    newspaper = rng.uniform(0, 100, size=n_samples)
    # 使用一个简单的线性生成规则并添加噪声
    sales = 3.0 + 0.045 * TV + 0.187 * radio + 0.001 * newspaper + rng.normal(0, 1.0, size=n_samples)

    df = pd.DataFrame({
        'TV': TV,
        'radio': radio,
        'newspaper': newspaper,
        'sales': sales
    })

    # 确保目录存在
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"已生成合成数据并保存到: {out_path}")
    return df

# 2. 数据预处理
def preprocess_data(df):
    """数据预处理，返回特征矩阵、目标变量和标准化器

    支持数据集中目标列名为 `sales` 或 `Sales`。
    """
    if df is None:
        return None, None, None

    # 假设数据集包含TV、radio、newspaper和sales列
    feature_cols = ['TV', 'radio', 'newspaper']
    X = df[feature_cols]
    y = df.get('sales') if 'sales' in df.columns else df.get('Sales')
    if y is None:
        raise KeyError("未找到目标列 'sales' 或 'Sales' 在数据中")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# 3. 模型训练与调优
def train_models(X, y):
    """训练多种回归模型并返回结果"""
    if X is None or y is None:
        return None
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 定义模型列表
    models = {
        "最小二乘回归": LinearRegression(),
        "岭回归": Ridge(),
        "Lasso回归": Lasso(),
        "弹性网络回归": ElasticNet()
    }
    
    # 定义超参数网格
    param_grids = {
        "最小二乘回归": {},
        "岭回归": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        "Lasso回归": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        "弹性网络回归": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
    }
    
    results = {}
    
    # 对每个模型进行训练和调优
    for name, model in models.items():
        print(f"\n训练{name}模型...")
        
        if name == "最小二乘回归":
            # 最小二乘回归无需调参
            model.fit(X_train, y_train)
            best_model = model
        else:
            # 使用网格搜索进行超参数调优
            grid_search = GridSearchCV(
                model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        
        # 预测
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # 评估
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 保存结果
        results[name] = {
            'model': best_model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"{name}模型评估结果:")
        print(f"训练集 MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    return results, X_train, X_test, y_train, y_test

# 4. 绘制性能比较图
def plot_performance_comparison(results):
    """绘制不同模型的性能比较图"""
    if results is None:
        return
    
    # 提取模型名称和性能指标
    model_names = list(results.keys())
    train_rmse = [results[name]['train_rmse'] for name in model_names]
    test_rmse = [results[name]['test_rmse'] for name in model_names]
    train_r2 = [results[name]['train_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制RMSE比较图
    x = np.arange(len(model_names))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, train_rmse, width, label='训练集')
    rects2 = axes[0].bar(x + width/2, test_rmse, width, label='测试集')
    
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('各模型RMSE比较')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].legend()
    
    # 添加RMSE数值标签
    def add_labels(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(axes[0], rects1)
    add_labels(axes[0], rects2)
    
    # 绘制R²比较图
    rects3 = axes[1].bar(x - width/2, train_r2, width, label='训练集')
    rects4 = axes[1].bar(x + width/2, test_r2, width, label='测试集')
    
    axes[1].set_ylabel('R² 分数')
    axes[1].set_title('各模型R²分数比较')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].legend()
    
    # 添加R²数值标签
    add_labels(axes[1], rects3)
    add_labels(axes[1], rects4)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    plt.show()

# 5. 绘制训练集与测试集折线图
def plot_prediction_comparison(results, X_test, y_test):
    """绘制预测值与真实值的折线图比较"""
    if results is None or X_test is None or y_test is None:
        return
    
    # 获取测试集索引（用于x轴）
    test_indices = range(len(y_test))
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制真实值
    plt.plot(test_indices, y_test, 'b-', label='真实销量', linewidth=2)
    
    # 绘制各模型预测值
    colors = ['r--', 'g--', 'm--', 'c--']
    for i, (name, result) in enumerate(results.items()):
        plt.plot(test_indices, result['y_test_pred'], colors[i], label=f'{name}预测值')
    
    plt.xlabel('样本索引')
    plt.ylabel('销量')
    plt.title('各模型在测试集上的预测值与真实值比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison.png', dpi=300)
    plt.show()

# 6. 绘制特征重要性图
def plot_feature_importance(results):
    """绘制各模型的特征重要性"""
    if results is None:
        return
    
    feature_names = ['TV', 'radio', 'newspaper']
    models_with_coef = ['最小二乘回归', '岭回归', 'Lasso回归']
    
    # 创建图表
    fig, axes = plt.subplots(1, len(models_with_coef), figsize=(15, 5))
    
    for i, name in enumerate(models_with_coef):
        if name in results:
            model = results[name]['model']
            coef = model.coef_
            
            axes[i].bar(feature_names, coef)
            axes[i].set_title(f'{name}特征重要性')
            axes[i].set_xlabel('特征')
            axes[i].set_ylabel('系数值')
            
            # 添加数值标签
            for j, v in enumerate(coef):
                axes[i].text(j, v, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

# 主函数
def main():
    # 使用仓库内相对 data 路径；如果不存在则生成合成数据
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'Advertising.csv')

    if not os.path.exists(file_path):
        print(f"未找到数据文件: {file_path}，将生成合成数据（N=100）。")
        generate_synthetic_data(file_path, n_samples=100)

    # 1. 加载数据
    df = load_data(file_path)
    
    # 2. 数据预处理
    X, y, scaler = preprocess_data(df)
    
    # 3. 训练模型
    if X is None or y is None:
        print("预处理失败，退出。")
        return

    results_tuple = train_models(X, y)
    if results_tuple is None:
        print("模型训练未能执行。")
        return
    results, X_train, X_test, y_train, y_test = results_tuple
    
    # 4. 绘制性能比较图
    plot_performance_comparison(results)
    
    # 5. 绘制预测值与真实值比较图
    plot_prediction_comparison(results, X_test, y_test)
    
    # 6. 绘制特征重要性图
    plot_feature_importance(results)
    
    # 7. 业务建议
    if not results:
        print("没有可用的模型结果用于推荐。")
        return

    best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
    best_model = results[best_model_name]['model']
    
    print("\n根据模型评估结果，推荐使用以下模型进行销量预测：")
    print(f"最佳模型: {best_model_name}")
    print(f"测试集RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"测试集R²分数: {results[best_model_name]['test_r2']:.4f}")
    
    if best_model_name in ['最小二乘回归', '岭回归', 'Lasso回归']:
        print("\n特征重要性（系数）:")
        coef_df = pd.DataFrame({
            '特征': ['TV', 'radio', 'newspaper'],
            '系数': best_model.coef_
        })
        print(coef_df.sort_values('系数', ascending=False))
        print("\n根据特征重要性，建议在广告投放中：")
        print("1. 增加系数为正且绝对值较大的特征的投入")
        print("2. 减少系数为负或绝对值较小的特征的投入")

if __name__ == "__main__":
    main()
