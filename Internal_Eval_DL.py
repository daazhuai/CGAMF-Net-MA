import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    f1_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
import os
import json
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================== 神经网络组件 ========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, z_k, z_ref):
        g = torch.sigmoid(self.fc(torch.cat([z_k, z_ref], dim=1)))
        return g * z_k


class MultiOmicNet(nn.Module):
    def __init__(self, dims, hidden, n_class):
        super().__init__()
        self.mlp = nn.ModuleDict({
            k: MLP(dims[k], hidden) for k in dims
        })
        self.gate = nn.ModuleDict({
            k: Gate(hidden) for k in dims if k != "clin"
        })
        self.classifier = nn.Linear(hidden, n_class)
        self.used_modalities = list(dims.keys())

    def forward(self, xs):
        z = {k: self.mlp[k](xs[k]) for k in self.used_modalities}

        if "clin" in z:
            z_ref = z["clin"]
            fused = z_ref
            for k in z:
                if k != "clin":
                    fused = fused + self.gate[k](z[k], z_ref)
        else:
            first_key = list(z.keys())[0]
            z_ref = z[first_key]
            fused = z_ref
            for k in z:
                if k != first_key:
                    fused = fused + self.gate[k](z[k], z_ref)

        return self.classifier(fused)


# ======================== 模型缓存管理 ========================
class ModelCache:
    def __init__(self, cache_dir="model_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, modalities, run_id, split_id=0):
        """生成缓存键"""
        modalities_str = "_".join(sorted(modalities))
        return f"run_{run_id}_split_{split_id}_{modalities_str}"

    def load_model(self, model_class, dims, hidden_dim, n_classes, modalities, run_id, split_id, device):
        """加载模型"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")

        if os.path.exists(cache_path):
            model = model_class(dims, hidden_dim, n_classes)
            model.load_state_dict(torch.load(cache_path))
            model.to(device)
            model.eval()
            return model
        return None

    def model_exists(self, modalities, run_id, split_id=0):
        """检查模型是否存在"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")
        return os.path.exists(cache_path)


# ======================== 评估函数（新增指标） ========================
def evaluate_predictions(probs, preds, true_labels, n_classes):
    """评估预测结果（包含新增指标）"""
    # 分类指标
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)

    # ROC AUC
    if n_classes == 2:
        roc_auc = roc_auc_score(true_labels, probs[:, 1])
    else:
        y_true_bin = label_binarize(true_labels, classes=range(n_classes))
        roc_auc_per_class = []
        for class_idx in range(n_classes):
            if np.sum(y_true_bin[:, class_idx]) > 0:
                class_roc_auc = roc_auc_score(y_true_bin[:, class_idx], probs[:, class_idx])
                roc_auc_per_class.append(class_roc_auc)

        if roc_auc_per_class:
            class_weights = [np.sum(true_labels == i) / len(true_labels) for i in range(len(roc_auc_per_class))]
            roc_auc = np.average(roc_auc_per_class, weights=class_weights)
        else:
            roc_auc = 0

    # PRAUC
    if n_classes == 2:
        prauc = average_precision_score(true_labels, probs[:, 1])
    else:
        prauc_scores = []
        for class_idx in range(n_classes):
            class_true = (true_labels == class_idx).astype(int)
            if np.sum(class_true) > 0:
                class_score = average_precision_score(class_true, probs[:, class_idx])
                prauc_scores.append(class_score)

        if prauc_scores:
            class_weights = [np.sum(true_labels == i) / len(true_labels) for i in range(len(prauc_scores))]
            prauc = np.average(prauc_scores, weights=class_weights)
        else:
            prauc = 0

    # === 新增评估指标 ===
    # Log Loss (交叉熵)
    try:
        logloss = log_loss(true_labels, probs)
    except:
        logloss = np.nan

    # MSE (均方误差) - 对概率预测计算
    y_true_onehot = np.eye(n_classes)[true_labels]
    mse = mean_squared_error(y_true_onehot, probs)

    # MAE (平均绝对误差) - 对概率预测计算
    mae = mean_absolute_error(y_true_onehot, probs)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'prauc': prauc,
        'log_loss': logloss,
        'mse': mse,
        'mae': mae
    }


# ======================== 基础网络评估函数 ========================
def evaluate_base_network_for_split(all_modalities, modality_combinations, X_data, y_data,
                                    n_classes, model_cache, run_id,
                                    split_id=0, random_seed=42):
    """
    对一次划分评估基础网络（单个网络，非集成）
    """
    set_seed(random_seed)

    # 1. 使用相同的随机种子划分数据
    total_indices = np.arange(len(y_data))
    train_idx, val_idx = train_test_split(
        total_indices,
        test_size=0.2,
        random_state=random_seed,
        stratify=y_data
    )

    # 2. 标准化数据
    train_X_dict = {}
    val_X_dict = {}

    for mod in all_modalities:
        X_mod = X_data[mod]

        scaler = StandardScaler()
        X_train = X_mod[train_idx]
        scaler.fit(X_train)

        X_train_std = scaler.transform(X_train)
        X_val_std = scaler.transform(X_mod[val_idx])

        train_X_dict[mod] = torch.tensor(X_train_std, dtype=torch.float32)
        val_X_dict[mod] = torch.tensor(X_val_std, dtype=torch.float32)

    val_y_np = y_data[val_idx]

    # 3. 评估每种模态组合的基础网络
    results = {}

    for combo_name, modalities in modality_combinations.items():
        # 检查模型是否存在
        if model_cache.model_exists(modalities, run_id, split_id):
            # 加载模型
            dims = {mod: X_data[mod].shape[1] for mod in modalities}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                           modalities, run_id, split_id, device)

            if model is not None:
                # 在验证集上预测
                model.eval()
                val_dataset = TensorDataset(*[val_X_dict[mod] for mod in modalities])
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                all_probs = []
                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_X_dict = {}
                        for i, mod in enumerate(modalities):
                            batch_X_dict[mod] = batch_data[i].to(device)

                        outputs = model(batch_X_dict)
                        probs = F.softmax(outputs, dim=1)
                        all_probs.extend(probs.cpu().numpy())

                probs_array = np.array(all_probs)
                pred_labels = np.argmax(probs_array, axis=1)

                # 评估（使用新增指标的评估函数）
                eval_metrics = evaluate_predictions(probs_array, pred_labels, val_y_np, n_classes)

                results[combo_name] = {
                    'probs': probs_array,
                    'labels': pred_labels,
                    'metrics': eval_metrics,
                    'true_labels': val_y_np.tolist()
                }
            else:
                print(f"警告: 无法加载模型 {combo_name} (run_id={run_id})")
                results[combo_name] = None
        else:
            print(f"警告: 模型文件不存在 {combo_name} (run_id={run_id})")
            results[combo_name] = None

    return results


# ======================== 主程序 ========================
def main():
    print("=" * 80)
    print("基础网络评估 (8种模态组合) - 包含Log Loss、MSE、MAE指标")
    print("=" * 80)

    # 设置随机种子
    set_seed(42)

    # 0. 数据准备
    print("\n0. 加载数据...")

    def read_csv_file(path):
        return pd.read_csv(path)

    # tcga
    # def normalize_sample_id(x):
    #     return str(x).split("-01")[0].split("-02")[0]

    # metabric
    def normalize_sample_id(x):
        return str(x)  # 或者根据实际格式处理

    # 文件路径（请根据实际情况修改）
    clinical_path = "./dataset/METABRIC/METABRIC_Clinical.csv"
    snv_path = "./dataset/METABRIC/METABRIC_SNV.csv"
    cnv_path = "./dataset/METABRIC/METABRIC_CNV.csv"
    mrna_path = "./dataset/METABRIC/METABRIC_RNA.csv"

    # 读取数据
    clin = read_csv_file(clinical_path)
    snv = read_csv_file(snv_path)
    cnv = read_csv_file(cnv_path)
    mrna = read_csv_file(mrna_path)

    def set_index_and_clean(df, is_clinical=False):
        df = df.copy()

        sample_id_columns = ['SAMPLE_ID', 'Sample_ID', 'sample_id', 'sample']
        sample_col = None

        for col in sample_id_columns:
            if col in df.columns:
                sample_col = col
                break

        if sample_col is None and len(df.columns) > 0:
            sample_col = df.columns[0]

        if sample_col:
            df[sample_col] = df[sample_col].astype(str).map(normalize_sample_id)
            df = df.set_index(sample_col)
            df = df.reset_index()
            df = df.set_index(sample_col)

        if is_clinical:
            categorical_cols = []
            numerical_cols = []

            for col in df.columns:
                if col == 'AGE' or col == 'SUBTYPE':
                    continue

                try:
                    pd.to_numeric(df[col], errors='raise')
                    numerical_cols.append(col)
                except:
                    categorical_cols.append(col)

            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except:
                    df = df.drop(columns=[col])

            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def clean_numeric_data(df, name):
        df = df.copy()

        non_numeric_cols = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
            except:
                non_numeric_cols.append(col)

        if non_numeric_cols:
            df = df.drop(columns=non_numeric_cols)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)

        return df

    # 处理各数据表
    clin = set_index_and_clean(clin, is_clinical=True)
    snv = set_index_and_clean(snv)
    cnv = set_index_and_clean(cnv)
    mrna = set_index_and_clean(mrna)

    # 清理特征数据
    snv = clean_numeric_data(snv, "SNV")
    cnv = clean_numeric_data(cnv, "CNV")
    mrna = clean_numeric_data(mrna, "mRNA")

    # 提取标签和临床特征
    if 'SUBTYPE' in clin.columns:
        y = clin['SUBTYPE']
        clin_feat = clin.drop(columns=['SUBTYPE'])
    else:
        raise ValueError("Clinical data must contain 'SUBTYPE' column")

    clin_feat = clean_numeric_data(clin_feat, "Clinical features")

    # 查找共同样本
    common_samples = list(
        set(snv.index) &
        set(cnv.index) &
        set(mrna.index) &
        set(clin_feat.index) &
        set(y.index)
    )

    print(f"匹配样本数量: {len(common_samples)}")

    # 筛选共同样本
    snv = snv.loc[common_samples]
    cnv = cnv.loc[common_samples]
    mrna = mrna.loc[common_samples]
    clin_feat = clin_feat.loc[common_samples]
    y = y.loc[common_samples]

    print("\n筛选后数据形状:")
    print(f"SNV: {snv.shape}")
    print(f"CNV: {cnv.shape}")
    print(f"mRNA: {mrna.shape}")
    print(f"Clinical features: {clin_feat.shape}")
    print(f"Labels: {y.shape}")

    # 转换为numpy数组
    X_snv_raw = snv.values.astype(np.float32)
    X_cnv_raw = cnv.values.astype(np.float32)
    X_mrna_raw = mrna.values.astype(np.float32)
    X_clin_raw = clin_feat.values.astype(np.float32)

    # 编码标签
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    n_classes = len(np.unique(y_enc))

    print(f"\n类别数量: {n_classes}")
    print(f"类别标签: {list(le_y.classes_)}")

    # 准备数据字典
    X_data = {
        "clin": X_clin_raw,
        "cnv": X_cnv_raw,
        "snv": X_snv_raw,
        "mrna": X_mrna_raw,
    }
    y_data = y_enc

    # 1. 定义所有模态和8种模态组合
    print("\n1. 定义评估组合...")

    # 所有可用模态
    all_modalities = ['clin', 'cnv', 'snv', 'mrna']

    # 定义8种模态组合
    modality_combinations = {
        'clinical': ['clin'],
        'clinical+CNV': ['clin', 'cnv'],
        'clinical+SNV': ['clin', 'snv'],
        'clinical+mRNA': ['clin', 'mrna'],
        'clinical+CNV+SNV': ['clin', 'cnv', 'snv'],
        'clinical+CNV+mRNA': ['clin', 'cnv', 'mrna'],
        'clinical+SNV+mRNA': ['clin', 'snv', 'mrna'],
        'clinical+CNV+SNV+mRNA': ['clin', 'cnv', 'snv', 'mrna']
    }

    print(f"所有可用模态: {all_modalities}")
    print(f"将评估 {len(modality_combinations)} 种模态组合:")
    for name, combo in modality_combinations.items():
        print(f"  {name}: {combo}")

    # 2. 设置缓存目录
    print("\n2. 设置模型缓存...")

    # 假设模型缓存在以下目录（请根据实际情况修改）
    # tcga
    # model_cache_dir = "/root/BRCA/model_cache"  # 使用缓存目录，模型已经训练好了
    # metabric
    model_cache_dir = "/root/METAtrain/model_cache"  # 使用缓存目录，模型已经训练好了
    model_cache = ModelCache(cache_dir=model_cache_dir)

    # 检查模型是否存在
    print(f"模型缓存目录: {model_cache_dir}")

    # 3. 创建输出目录
    print("\n3. 创建输出目录...")

    output_dirs = {
        'predictions': './base_predictions',
        'evaluations': './dl_evaluations',
        'summary': './base_summary'
    }

    for dir_name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_path}")

    # 4. 进行100次划分评估
    print("\n4. 开始100次划分评估...")
    start_time = time.time()

    n_runs = 100
    all_run_results = {}

    # 更新指标列表，包含新增的评估指标
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'prauc', 'log_loss', 'mse', 'mae']

    # 进度条
    pbar = tqdm(range(n_runs), desc="评估进度")

    for run_id in pbar:
        random_seed = 42 + run_id

        # 对这次划分评估基础网络
        run_result = evaluate_base_network_for_split(
            all_modalities=all_modalities,
            modality_combinations=modality_combinations,
            X_data=X_data,
            y_data=y_data,
            n_classes=n_classes,
            model_cache=model_cache,
            run_id=run_id,
            split_id=0,
            random_seed=random_seed
        )

        # 保存预测结果（包含所有指标）
        for combo_name, combo_results in run_result.items():
            if combo_results is not None:
                pred_data = {
                    'probs': combo_results['probs'],
                    'labels': combo_results['labels'],
                    'true_labels': combo_results['true_labels'],
                    'metrics': combo_results['metrics']  # 现在包含9个指标
                }

                pred_file = os.path.join(
                    output_dirs['predictions'],
                    f"base_pred_run_{run_id:03d}_{combo_name}.pkl"
                )
                with open(pred_file, 'wb') as f:
                    pickle.dump(pred_data, f)

        # 收集评估结果
        for combo_name, combo_results in run_result.items():
            if combo_results is not None:
                if combo_name not in all_run_results:
                    all_run_results[combo_name] = {metric: [] for metric in metrics_list}

                metrics = combo_results['metrics']
                for metric_name in metrics.keys():
                    if metric_name in all_run_results[combo_name]:
                        all_run_results[combo_name][metric_name].append(metrics[metric_name])

        # 更新进度条
        pbar.set_postfix({
            '当前运行': run_id + 1,
            '总运行': n_runs
        })

    elapsed_time = time.time() - start_time
    print(f"\n所有评估完成，总耗时: {elapsed_time:.2f} 秒")

    # 5. 保存评估结果
    print("\n5. 保存评估结果...")

    # 更新指标显示名称，包含新增指标
    metric_displays = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'PR_AUC',
                       'Log_Loss', 'MSE', 'MAE']

    # 创建汇总表格
    table_data = []

    for metric_display in metric_displays:
        metric_name = metric_display.lower().replace('_', '').replace('-', '')
        if metric_name == 'accuracy':
            metric_name = 'accuracy'
        elif metric_name == 'precision':
            metric_name = 'precision'
        elif metric_name == 'recall':
            metric_name = 'recall'
        elif metric_name == 'f1score':
            metric_name = 'f1'
        elif metric_name == 'rocauc':
            metric_name = 'roc_auc'
        elif metric_name == 'prauc':
            metric_name = 'prauc'
        elif metric_name == 'logloss':
            metric_name = 'log_loss'
        elif metric_name == 'mse':
            metric_name = 'mse'
        elif metric_name == 'mae':
            metric_name = 'mae'

        row = {'Metric': metric_display}

        for combo_name in modality_combinations.keys():
            if (combo_name in all_run_results and
                    metric_name in all_run_results[combo_name]):
                values = all_run_results[combo_name][metric_name]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    max_val = np.max(values)
                    min_val = np.min(values)

                    # 格式：均值±标准差 [最小值, 最大值]
                    cell_value = f"{mean_val:.4f}±{std_val:.4f} [{min_val:.4f}, {max_val:.4f}]"
                    row[combo_name] = cell_value
                else:
                    row[combo_name] = "N/A"
            else:
                row[combo_name] = "N/A"

        table_data.append(row)

    # 创建DataFrame
    df_table = pd.DataFrame(table_data)

    # 保存CSV
    csv_file = os.path.join(output_dirs['evaluations'], "base_network_summary_100runs.csv")
    df_table.to_csv(csv_file, index=False)

    # 保存详细结果
    detailed_results = {}
    for combo_name in modality_combinations.keys():
        if combo_name in all_run_results:
            detailed_results[combo_name] = {}
            for metric_name in metrics_list:
                if metric_name in all_run_results[combo_name]:
                    values = all_run_results[combo_name][metric_name]
                    if len(values) > 0:
                        detailed_results[combo_name][metric_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'max': float(np.max(values)),
                            'min': float(np.min(values)),
                            'values': [float(v) for v in values]
                        }

    detailed_file = os.path.join(output_dirs['evaluations'], "base_network_detailed_results.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n基础网络结果已保存:")
    print(f"  - 汇总表格: {csv_file}")
    print(f"  - 详细结果: {detailed_file}")

    # 6. 创建最佳模型比较表格（更新指标列表）
    print("\n6. 创建最佳模型比较表格...")

    # 找出每种模态组合的最佳指标
    best_comparison = []
    for metric_name, metric_display in zip(metrics_list, metric_displays):
        best_value = -float('inf') if metric_name not in ['log_loss', 'mse', 'mae'] else float('inf')
        best_model = ""

        for combo_name in modality_combinations.keys():
            if combo_name in all_run_results and metric_name in all_run_results[combo_name]:
                values = all_run_results[combo_name][metric_name]
                if len(values) > 0:
                    mean_val = np.mean(values)

                    # 对于损失类指标，越小越好；对于性能指标，越大越好
                    if metric_name in ['log_loss', 'mse', 'mae']:
                        if mean_val < best_value:
                            best_value = mean_val
                            best_model = combo_name
                    else:
                        if mean_val > best_value:
                            best_value = mean_val
                            best_model = combo_name

        best_comparison.append({
            'Metric': metric_display,
            'Best Model': best_model,
            'Best Value': f"{best_value:.4f}"
        })

    df_best = pd.DataFrame(best_comparison)
    best_file = os.path.join(output_dirs['summary'], "base_network_best_models.csv")
    df_best.to_csv(best_file, index=False)
    print(f"  最佳模型比较: {best_file}")

    # 7. 打印总结
    print("\n" + "=" * 80)
    print("评估完成总结")
    print("=" * 80)

    print(f"\n输出文件结构:")
    print(f"1. 预测结果目录: {output_dirs['predictions']}")
    print(f"   包含800个文件: base_pred_run_000_*.pkl")
    print(f"   每个文件包含预测概率、预测标签、真实标签和9个评估指标")

    print(f"\n2. 评估结果目录: {output_dirs['evaluations']}")
    print(f"   包含2个文件:")
    print(f"   - base_network_summary_100runs.csv (包含9个指标)")
    print(f"   - base_network_detailed_results.json (包含9个指标的详细统计)")

    print(f"\n3. 综合比较目录: {output_dirs['summary']}")
    print(f"   包含1个最佳模型比较文件 (包含9个指标的最佳模型)")

    print(f"\n实验配置:")
    print(f"- 随机划分次数: {n_runs}")
    print(f"- 评估指标数量: 9 (新增: Log Loss, MSE, MAE)")
    print(f"- 模态组合数量: {len(modality_combinations)}")
    print(f"- 总样本数: {len(y_data)}")
    print(f"- 类别数量: {n_classes}")
    print(f"- 总耗时: {elapsed_time:.2f} 秒")

    print(f"\n评估的模态组合:")
    for name, combo in modality_combinations.items():
        print(f"  {name}: {combo}")

    print("\n新增评估指标说明:")
    print("- Log Loss: 对数损失，衡量概率预测的准确性，越小越好")
    print("- MSE: 均方误差，衡量概率预测与真实标签的误差，越小越好")
    print("- MAE: 平均绝对误差，衡量概率预测与真实标签的绝对误差，越小越好")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()