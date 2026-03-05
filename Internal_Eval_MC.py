"""
使用AIC/BIC准则选择最优子模型
对8种模态组合，分别从所有子模型中选出AIC/BIC最优的模型进行评估
"""

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
from itertools import combinations
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
            model.load_state_dict(torch.load(cache_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        return None

    def model_exists(self, modalities, run_id, split_id=0):
        """检查模型是否存在"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")
        return os.path.exists(cache_path)


# ======================== 评估函数 ========================
def evaluate_predictions(probs, preds, true_labels, n_classes):
    """评估预测结果"""
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

    # Log Loss (交叉熵)
    try:
        logloss = log_loss(true_labels, probs)
    except:
        logloss = np.nan

    # MSE (均方误差)
    y_true_onehot = np.eye(n_classes)[true_labels]
    mse = mean_squared_error(y_true_onehot, probs)

    # MAE (平均绝对误差)
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


# ======================== AIC/BIC计算函数 ========================
def calculate_nll(y_true, y_pred_probs):
    """计算负对数似然"""
    y_pred_probs = np.clip(y_pred_probs, 1e-10, 1 - 1e-10)

    nll = 0
    for i in range(len(y_true)):
        true_class = int(y_true[i])
        nll -= np.log(y_pred_probs[i, true_class])

    return nll


def calculate_aic(n_samples, nll, feature_counts_dict, sub_modalities):
    """计算AIC值（使用特征数量）"""
    # 计算使用的特征总数
    total_features = 0
    for modality in sub_modalities:
        total_features += feature_counts_dict.get(modality, 0)

    # AIC = 2k + 2*nll
    return 2 * total_features + 2 * nll


def calculate_bic(n_samples, nll, feature_counts_dict, sub_modalities):
    """计算BIC值（使用特征数量）"""
    # 计算使用的特征总数
    total_features = 0
    for modality in sub_modalities:
        total_features += feature_counts_dict.get(modality, 0)

    # BIC = k*ln(n) + 2*nll
    return np.log(n_samples) * total_features + 2 * nll


# ======================== 主函数：AIC/BIC模型选择 ========================
def main():
    print("=" * 80)
    print("AIC/BIC模型选择评估")
    print("对8种模态组合，分别从子模型中选出AIC/BIC最优的模型")
    print("=" * 80)

    # 设置随机种子
    set_seed(42)

    # 0. 数据准备（与原始代码一致）
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

    # 获取特征数量
    feature_counts_dict = {}
    for mod, data in X_data.items():
        feature_counts_dict[mod] = data.shape[1]

    print(f"\n各模态特征数量: {feature_counts_dict}")

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

    # 2. 设置缓存目录
    print("\n2. 设置模型缓存...")
    model_cache_dir = "/root/METAtrain/model_cache"
    model_cache = ModelCache(cache_dir=model_cache_dir)
    print(f"模型缓存目录: {model_cache_dir}")

    # 3. 创建输出目录
    print("\n3. 创建输出目录...")
    output_dirs = {
        'predictions': './aic_bic_predict_results',  # 预测结果
        'best_models': './aic_bic_best_models',      # 最优模型记录
        'evaluations': './aic_bic_eva_results'       # 评估结果
    }

    for dir_name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_path}")

    # 4. 进行100次划分评估
    print("\n4. 开始100次划分的AIC/BIC模型选择评估...")
    start_time = time.time()

    n_runs = 100
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'prauc', 'log_loss', 'mse', 'mae']

    # 存储所有运行的结果
    all_run_results = {
        'AIC': {combo_name: {metric: [] for metric in metrics_list} for combo_name in modality_combinations.keys()},
        'BIC': {combo_name: {metric: [] for metric in metrics_list} for combo_name in modality_combinations.keys()}
    }

    # 存储最优模型记录（100次 * 8种组合 * 2种准则）
    best_models_records = []  # 将用于创建DataFrame

    # 进度条
    pbar = tqdm(range(n_runs), desc="AIC/BIC模型选择评估进度")

    for run_id in pbar:
        random_seed = 42 + run_id

        # 生成所有子模型组合
        all_sub_models = []
        for r in range(1, len(all_modalities) + 1):
            for combo in combinations(all_modalities, r):
                all_sub_models.append(list(combo))

        # 使用相同的随机种子划分数据
        total_indices = np.arange(len(y_data))
        train_idx, val_idx = train_test_split(
            total_indices,
            test_size=0.2,
            random_state=random_seed,
            stratify=y_data
        )

        # 标准化数据
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

        train_y_np = y_data[train_idx]
        val_y_np = y_data[val_idx]
        n_samples = len(train_idx)

        # 对每种模态组合进行AIC/BIC模型选择
        for combo_name, modalities in modality_combinations.items():
            # 找出该组合下的所有子模型
            combo_sub_models = []
            for sub_modalities in all_sub_models:
                if set(sub_modalities).issubset(set(modalities)):
                    combo_sub_models.append(tuple(sub_modalities))

            if not combo_sub_models:
                continue

            # 检查哪些子模型存在
            valid_sub_models = []
            for sub_modalities in combo_sub_models:
                if model_cache.model_exists(sub_modalities, run_id, split_id=0):
                    valid_sub_models.append(sub_modalities)

            if len(valid_sub_models) == 0:
                print(f"  警告: 组合 {combo_name} 没有可用的子模型")
                continue

            # 计算每个子模型的NLL和AIC/BIC
            sub_model_nll = {}
            sub_model_aic = {}
            sub_model_bic = {}

            for sub_modalities in valid_sub_models:
                # 加载模型
                dims = {mod: X_data[mod].shape[1] for mod in sub_modalities}
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                               sub_modalities, run_id, 0, device)

                if model is not None:
                    # 在训练集上预测（用于计算NLL）
                    model.eval()
                    train_dataset = TensorDataset(*[train_X_dict[mod] for mod in sub_modalities])
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

                    all_probs = []
                    with torch.no_grad():
                        for batch_data in train_loader:
                            batch_X_dict = {}
                            for i, mod in enumerate(sub_modalities):
                                batch_X_dict[mod] = batch_data[i].to(device)

                            outputs = model(batch_X_dict)
                            probs = F.softmax(outputs, dim=1)
                            all_probs.extend(probs.cpu().numpy())

                    probs_array = np.array(all_probs)

                    # 计算NLL
                    nll = calculate_nll(train_y_np, probs_array)

                    # 计算AIC和BIC
                    aic = calculate_aic(n_samples, nll, feature_counts_dict, sub_modalities)
                    bic = calculate_bic(n_samples, nll, feature_counts_dict, sub_modalities)

                    sub_model_nll[sub_modalities] = nll
                    sub_model_aic[sub_modalities] = aic
                    sub_model_bic[sub_modalities] = bic

            # 如果没有有效的子模型，跳过
            if len(sub_model_nll) == 0:
                continue

            # ========== AIC模型选择 ==========
            # 选择AIC最小的模型
            best_aic_model = min(sub_model_aic.items(), key=lambda x: x[1])[0]
            best_aic_value = sub_model_aic[best_aic_model]

            # ========== BIC模型选择 ==========
            # 选择BIC最小的模型
            best_bic_model = min(sub_model_bic.items(), key=lambda x: x[1])[0]
            best_bic_value = sub_model_bic[best_bic_model]

            # 记录最优模型信息
            best_models_records.append({
                'run_id': run_id,
                'combo_name': combo_name,
                'modalities': str(modalities),
                'criterion': 'AIC',
                'best_model': str(list(best_aic_model)),
                'best_value': best_aic_value
            })

            best_models_records.append({
                'run_id': run_id,
                'combo_name': combo_name,
                'modalities': str(modalities),
                'criterion': 'BIC',
                'best_model': str(list(best_bic_model)),
                'best_value': best_bic_value
            })

            # ========== 评估最优模型在验证集上的表现 ==========
            for criterion, best_model in [('AIC', best_aic_model), ('BIC', best_bic_model)]:
                # 加载最优模型
                dims = {mod: X_data[mod].shape[1] for mod in best_model}
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_model_obj = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                                        best_model, run_id, 0, device)

                if best_model_obj is not None:
                    # 在验证集上预测
                    best_model_obj.eval()
                    val_dataset = TensorDataset(*[val_X_dict[mod] for mod in best_model])
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                    all_probs = []
                    with torch.no_grad():
                        for batch_data in val_loader:
                            batch_X_dict = {}
                            for i, mod in enumerate(best_model):
                                batch_X_dict[mod] = batch_data[i].to(device)

                            outputs = best_model_obj(batch_X_dict)
                            probs = F.softmax(outputs, dim=1)
                            all_probs.extend(probs.cpu().numpy())

                    probs_array = np.array(all_probs)
                    pred_labels = np.argmax(probs_array, axis=1)

                    # 评估
                    eval_metrics = evaluate_predictions(probs_array, pred_labels, val_y_np, n_classes)

                    # 保存预测结果
                    pred_data = {
                        'probs': probs_array,
                        'labels': pred_labels,
                        'true_labels': val_y_np,
                        'best_model': list(best_model),
                        'metrics': eval_metrics
                    }

                    pred_file = os.path.join(
                        output_dirs['predictions'],
                        f"run_{run_id:03d}_{combo_name}_{criterion}.pkl"
                    )
                    with open(pred_file, 'wb') as f:
                        pickle.dump(pred_data, f)

                    # 收集评估结果
                    for metric_name in metrics_list:
                        if metric_name in eval_metrics:
                            all_run_results[criterion][combo_name][metric_name].append(eval_metrics[metric_name])

        # 更新进度条
        pbar.set_postfix({
            '当前运行': run_id + 1,
            '总运行': n_runs
        })

    elapsed_time = time.time() - start_time
    print(f"\n所有评估完成，总耗时: {elapsed_time:.2f} 秒")

    # 5. 保存最优模型记录
    print("\n5. 保存最优模型记录...")
    df_best_models = pd.DataFrame(best_models_records)
    best_models_file = os.path.join(output_dirs['best_models'], 'best_models_records.csv')
    df_best_models.to_csv(best_models_file, index=False)
    print(f"最优模型记录已保存: {best_models_file}")
    print(f"记录数量: {len(df_best_models)} (100次 * 8种组合 * 2种准则 = 1600条)")

    # 创建每种组合的统计信息
    print("\n  生成最优模型统计...")
    stats_data = []
    for combo_name in modality_combinations.keys():
        combo_stats = {'combo_name': combo_name}
        for criterion in ['AIC', 'BIC']:
            combo_data = df_best_models[(df_best_models['combo_name'] == combo_name) &
                                        (df_best_models['criterion'] == criterion)]
            if len(combo_data) > 0:
                # 统计各子模型被选中的次数
                model_counts = combo_data['best_model'].value_counts()
                combo_stats[f'{criterion}_most_frequent'] = model_counts.index[0] if len(model_counts) > 0 else 'N/A'
                combo_stats[f'{criterion}_most_freq_count'] = model_counts.iloc[0] if len(model_counts) > 0 else 0
                combo_stats[f'{criterion}_unique_models'] = len(model_counts)
            else:
                combo_stats[f'{criterion}_most_frequent'] = 'N/A'
                combo_stats[f'{criterion}_most_freq_count'] = 0
                combo_stats[f'{criterion}_unique_models'] = 0
        stats_data.append(combo_stats)

    df_stats = pd.DataFrame(stats_data)
    stats_file = os.path.join(output_dirs['best_models'], 'best_models_statistics.csv')
    df_stats.to_csv(stats_file, index=False)
    print(f"最优模型统计已保存: {stats_file}")

    # 6. 保存评估结果
    print("\n6. 保存评估结果...")

    metric_displays = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'PR_AUC',
                       'Log_Loss', 'MSE', 'MAE']

    for criterion in ['AIC', 'BIC']:
        # 为每种准则创建汇总表格
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
                if combo_name in all_run_results[criterion] and metric_name in all_run_results[criterion][combo_name]:
                    values = all_run_results[criterion][combo_name][metric_name]
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
        csv_file = os.path.join(output_dirs['evaluations'], f"{criterion}_summary_100runs.csv")
        df_table.to_csv(csv_file, index=False)

        # 保存详细结果
        detailed_results = {}
        for combo_name in modality_combinations.keys():
            if combo_name in all_run_results[criterion]:
                detailed_results[combo_name] = {}
                for metric_name in metrics_list:
                    if metric_name in all_run_results[criterion][combo_name]:
                        values = all_run_results[criterion][combo_name][metric_name]
                        if len(values) > 0:
                            detailed_results[combo_name][metric_name] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'max': float(np.max(values)),
                                'min': float(np.min(values)),
                                'values': [float(v) for v in values]
                            }

        detailed_file = os.path.join(output_dirs['evaluations'], f"{criterion}_detailed_results.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"  {criterion}方法结果已保存:")
        print(f"    - 汇总表格: {csv_file}")
        print(f"    - 详细结果: {detailed_file}")

    # 7. 打印总结
    print("\n" + "=" * 80)
    print("AIC/BIC模型选择评估完成总结")
    print("=" * 80)

    print(f"\n输出文件结构:")
    print(f"1. 预测结果目录: {output_dirs['predictions']}")
    print(f"   包含1600个文件: run_000_*_AIC.pkl, run_000_*_BIC.pkl 等")

    print(f"\n2. 最优模型记录目录: {output_dirs['best_models']}")
    print(f"   - best_models_records.csv (1600条记录)")
    print(f"   - best_models_statistics.csv (各组合统计)")

    print(f"\n3. 评估结果目录: {output_dirs['evaluations']}")
    print(f"   - AIC_summary_100runs.csv")
    print(f"   - AIC_detailed_results.json")
    print(f"   - BIC_summary_100runs.csv")
    print(f"   - BIC_detailed_results.json")

    print(f"\n实验配置:")
    print(f"- 随机划分次数: {n_runs}")
    print(f"- 模型选择准则: AIC, BIC")
    print(f"- 评估指标: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, MSE, MAE")
    print(f"- 模态组合数量: {len(modality_combinations)}")
    print(f"- 总耗时: {elapsed_time:.2f} 秒")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()