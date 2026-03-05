import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
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


def calculate_smoothed_ic_weights(ic_values):
    """计算光滑IC权重"""
    # w_m = exp{-IC_m/2} / Σ exp{-IC_j/2}

    # 第一步：计算 -IC/2
    neg_ic_half = -ic_values / 2

    # 第二步：找到最大值（用于数值稳定）
    max_val = np.max(neg_ic_half)

    # 第三步：计算 exp(-IC/2 - max_val) 避免数值下溢
    exp_vals = np.exp(neg_ic_half - max_val)

    # 第四步：计算分母
    sum_exp = np.sum(exp_vals)

    # 第五步：计算权重
    weights = exp_vals / sum_exp

    return weights


def solve_quadratic_program(Q_matrix):
    """使用二次规划求解CV权重"""
    try:
        from cvxopt import matrix, solvers

        n_models = Q_matrix.shape[0]

        # 转换为cvxopt格式
        P = matrix(Q_matrix.astype(float))
        q = matrix(np.zeros(n_models))

        # 约束条件: w_i >= 0
        G = matrix(-np.eye(n_models))
        h = matrix(np.zeros(n_models))

        # 约束条件: sum(w) = 1
        A = matrix(np.ones((1, n_models)).astype(float))
        b = matrix(1.0)

        # 求解
        solvers.options['show_progress'] = False
        solvers.options['abstol'] = 1e-10
        solvers.options['reltol'] = 1e-10
        solvers.options['feastol'] = 1e-10

        solution = solvers.qp(P, q, G, h, A, b)

        if solution['status'] == 'optimal':
            weights = np.array(solution['x']).flatten()
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_models) / n_models
            return weights
        else:
            print(f"优化未收敛: {solution['status']}")
            return np.ones(n_models) / n_models

    except ImportError:
        # 使用SciPy替代
        from scipy.optimize import minimize

        n_models = Q_matrix.shape[0]

        def objective(w):
            return 0.5 * w @ Q_matrix @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_models)]
        initial_guess = np.ones(n_models) / n_models

        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})

        weights = np.maximum(result.x, 0)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else initial_guess
        return weights


# ======================== 主要评估函数 ========================
def evaluate_one_split_with_methods(all_modalities, modality_combinations, X_data, y_data,
                                    feature_counts_dict, n_classes, model_cache, run_id,
                                    split_id=0, random_seed=42):
    """
    对一次划分应用三种权重选取方法
    """
    set_seed(random_seed)

    # 1. 生成所有子模型组合
    all_sub_models = []
    for r in range(1, len(all_modalities) + 1):
        for combo in combinations(all_modalities, r):
            all_sub_models.append(list(combo))

    # 2. 使用相同的随机种子划分数据
    total_indices = np.arange(len(y_data))
    train_idx, val_idx = train_test_split(
        total_indices,
        test_size=0.2,
        random_state=random_seed,
        stratify=y_data
    )

    # 3. 标准化数据
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

    # 4. 加载所有子模型并获取预测
    sub_model_predictions_val = {}
    sub_model_nll = {}
    sub_model_modalities = {}

    for sub_model_idx, sub_modalities in enumerate(all_sub_models):
        # 检查模型是否存在
        if model_cache.model_exists(sub_modalities, run_id, split_id):
            # 加载模型
            dims = {mod: X_data[mod].shape[1] for mod in sub_modalities}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                           sub_modalities, run_id, split_id, device)

            if model is not None:
                # 在验证集上预测
                model.eval()
                val_dataset = TensorDataset(*[val_X_dict[mod] for mod in sub_modalities])
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                all_probs = []
                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_X_dict = {}
                        for i, mod in enumerate(sub_modalities):
                            batch_X_dict[mod] = batch_data[i].to(device)

                        outputs = model(batch_X_dict)
                        probs = F.softmax(outputs, dim=1)
                        all_probs.extend(probs.cpu().numpy())

                probs_array = np.array(all_probs)
                sub_model_predictions_val[tuple(sub_modalities)] = probs_array
                sub_model_nll[tuple(sub_modalities)] = calculate_nll(val_y_np, probs_array)
                sub_model_modalities[tuple(sub_modalities)] = sub_modalities

    # 5. 为每种模态组合计算四种方法的权重（CV, AIC, BIC, Equal）
    results = {}

    for combo_name, modalities in modality_combinations.items():
        # 找出该组合下的所有子模型
        combo_sub_models = []
        for sub_modalities in all_sub_models:
            if set(sub_modalities).issubset(set(modalities)):
                combo_sub_models.append(tuple(sub_modalities))

        if not combo_sub_models:
            continue

        # 检查是否所有子模型都有预测结果
        valid_combo_sub_models = []
        for sub_modalities in combo_sub_models:
            if sub_modalities in sub_model_predictions_val:
                valid_combo_sub_models.append(sub_modalities)

        if len(valid_combo_sub_models) == 0:
            continue

        n_sub_models = len(valid_combo_sub_models)

        # ===================== CV权重计算（使用K折交叉验证）=====================
        # 在训练集上进行K折交叉验证来获取CV权重
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=random_seed)

        # 存储每折的预测结果
        cv_train_predictions = np.zeros((K, n_sub_models, len(train_idx), n_classes))

        print(f"  为组合 {combo_name} 进行{K}折CV权重计算...")

        for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kf.split(train_idx)):
            # 获取当前折的训练和验证索引
            train_fold_actual = train_idx[train_fold_idx]
            val_fold_actual = train_idx[val_fold_idx]

            # 标准化（重新拟合当前折的训练数据）
            fold_train_X_dict = {}
            fold_val_X_dict = {}

            for mod in all_modalities:
                X_mod = X_data[mod]
                scaler = StandardScaler()
                X_fold_train = X_mod[train_fold_actual]
                scaler.fit(X_fold_train)

                fold_train_X_dict[mod] = torch.tensor(scaler.transform(X_fold_train), dtype=torch.float32)
                fold_val_X_dict[mod] = torch.tensor(scaler.transform(X_mod[val_fold_actual]), dtype=torch.float32)

            # 获取当前折的验证集预测
            fold_val_y = y_data[val_fold_actual]

            # 为每个子模型获取预测
            for model_idx, sub_modalities in enumerate(valid_combo_sub_models):
                # 加载模型
                dims = {mod: X_data[mod].shape[1] for mod in sub_modalities}
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                               sub_modalities, run_id, split_id, device)

                if model is not None:
                    model.eval()
                    fold_val_dataset = TensorDataset(*[fold_val_X_dict[mod] for mod in sub_modalities])
                    fold_val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False)

                    all_probs = []
                    with torch.no_grad():
                        for batch_data in fold_val_loader:
                            batch_X_dict = {}
                            for i, mod in enumerate(sub_modalities):
                                batch_X_dict[mod] = batch_data[i].to(device)

                            outputs = model(batch_X_dict)
                            probs = F.softmax(outputs, dim=1)
                            all_probs.extend(probs.cpu().numpy())

                    # 将预测结果存储到正确位置
                    probs_array = np.array(all_probs)
                    cv_train_predictions[fold_idx, model_idx, val_fold_idx, :] = probs_array

        # 使用所有折叠的预测结果来优化CV权重
        # 准备数据：将所有折叠的预测拼接起来
        cv_all_predictions = np.zeros((n_sub_models, len(train_idx), n_classes))
        for model_idx in range(n_sub_models):
            # 从每个折叠中提取该模型的预测
            for fold_idx in range(K):
                _, val_fold_idx = list(kf.split(train_idx))[fold_idx]
                cv_all_predictions[model_idx, val_fold_idx, :] = cv_train_predictions[fold_idx, model_idx, val_fold_idx,
                                                                 :]

        # 使用平方损失（MSE）计算CV权重
        # 创建one-hot编码的真实标签
        y_train_onehot = np.eye(n_classes)[train_y_np]  # shape: (n_train, n_classes)

        # 计算每个模型的预测误差（连续值）
        errors_continuous = np.zeros((n_sub_models, len(train_idx), n_classes))
        for m in range(n_sub_models):
            errors_continuous[m] = cv_all_predictions[m] - y_train_onehot

        # 展平误差矩阵：将每个样本的每个类别视为一个独立的误差项
        errors_flat = errors_continuous.reshape(n_sub_models, -1)  # shape: (n_sub_models, n_train * n_classes)

        # 计算Q矩阵
        Q = (errors_flat @ errors_flat.T) / (len(train_idx) * n_classes)
        cv_weights = solve_quadratic_program(Q)

        # ===================== AIC权重计算 =====================
        # 使用完整的训练集计算AIC
        aic_train_predictions = {}
        for sub_modalities in valid_combo_sub_models:
            # 加载模型
            dims = {mod: X_data[mod].shape[1] for mod in sub_modalities}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes,
                                           sub_modalities, run_id, split_id, device)

            if model is not None:
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
                aic_train_predictions[sub_modalities] = probs_array

        n_samples = len(train_idx)
        aic_values = []
        for sub_modalities in valid_combo_sub_models:
            if sub_modalities in aic_train_predictions:
                nll = calculate_nll(train_y_np, aic_train_predictions[sub_modalities])
                aic = calculate_aic(n_samples, nll, feature_counts_dict, sub_modalities)
                aic_values.append(aic)
            else:
                aic_values.append(np.inf)  # 如果无法获取预测，使用无穷大

        aic_weights = calculate_smoothed_ic_weights(np.array(aic_values))

        # ===================== BIC权重计算 =====================
        bic_values = []
        for sub_modalities in valid_combo_sub_models:
            if sub_modalities in aic_train_predictions:
                nll = calculate_nll(train_y_np, aic_train_predictions[sub_modalities])
                bic = calculate_bic(n_samples, nll, feature_counts_dict, sub_modalities)
                bic_values.append(bic)
            else:
                bic_values.append(np.inf)

        bic_weights = calculate_smoothed_ic_weights(np.array(bic_values))

        # ===================== Equal权重计算（新增）=====================
        equal_weights = np.ones(n_sub_models) / n_sub_models

        # 保存权重
        weights_data = {
            'CV': cv_weights.tolist(),
            'AIC': aic_weights.tolist(),
            'BIC': bic_weights.tolist(),
            'Equal': equal_weights.tolist(),  # 新增平均权重
            'sub_models': [list(m) for m in valid_combo_sub_models]
        }

        # 计算加权平均预测（在验证集上）
        predictions = {}
        # 修改：增加 Equal 方法
        for method, weights in [('CV', cv_weights), ('AIC', aic_weights), ('BIC', bic_weights), ('Equal', equal_weights)]:
            final_predictions = np.zeros((len(val_idx), n_classes))
            for idx, sub_modalities in enumerate(valid_combo_sub_models):
                final_predictions += weights[idx] * sub_model_predictions_val[sub_modalities]

            final_pred_labels = np.argmax(final_predictions, axis=1)

            # 评估
            eval_metrics = evaluate_predictions(final_predictions, final_pred_labels, val_y_np, n_classes)

            predictions[method] = {
                'probs': final_predictions,
                'labels': final_pred_labels,
                'metrics': eval_metrics
            }

        results[combo_name] = {
            'ma_weights': weights_data,
            'predictions': predictions,
            'true_labels': val_y_np.tolist()
        }

    return results


# ======================== 主程序 ========================
def main():
    print("=" * 80)
    print("四种权重选取方法比较评估 (CV, AIC, BIC, Equal)")
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
        return str(x)  # 或者根据需要处理

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

    # 在标签编码后添加这个代码段
    print("\n=== 类别标签详细映射 ===")
    print(f"类别标签数组: {list(le_y.classes_)}")
    print(f"类别数量: {len(le_y.classes_)}")

    # 打印前10个样本的原始标签和编码
    print("\n前10个样本的标签:")
    for i in range(min(10, len(y))):
        print(f"  样本 {i}: {y.iloc[i]} -> 编码: {y_enc[i]}")

    # 创建完整的映射字典
    label_mapping = dict(zip(range(len(le_y.classes_)), le_y.classes_))
    print(f"\n类别编码映射: {label_mapping}")

    # 保存映射关系供可视化使用
    import json
    label_mapping_dict = {f"Class {i + 1}": label_mapping[i] for i in range(len(label_mapping))}
    with open('class_label_mapping.json', 'w') as f:
        json.dump(label_mapping_dict, f, indent=2)
    print(f"映射已保存到: class_label_mapping.json")

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
    for name, combo in modality_combinations.items():
        n_sub_models = 2 ** len(combo) - 1
        print(f"  {name}: {combo} (将使用 {n_sub_models} 个子模型)")

    # 2. 设置缓存目录
    print("\n2. 设置模型缓存...")

    # 假设模型缓存在以下目录（请根据实际情况修改）
    model_cache_dir = "/root/METAtrain/model_cache"  # 使用缓存目录，模型已经训练好了
    model_cache = ModelCache(cache_dir=model_cache_dir)

    # 检查模型是否存在
    print(f"模型缓存目录: {model_cache_dir}")

    # 3. 创建输出目录
    print("\n3. 创建输出目录...")

    output_dirs = {
        'ma_weights': './ma_weights',
        'predictions': './ma_predict_result',
        'evaluations': './eva_result',
        'summary': './summary'
    }

    for dir_name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_path}")

    # 4. 进行100次划分评估
    print("\n4. 开始100次划分评估...")
    start_time = time.time()

    n_runs = 100
    all_run_results = {}

    # 进度条
    pbar = tqdm(range(n_runs), desc="评估进度")

    # 更新指标列表，包含新增的评估指标
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'prauc', 'log_loss', 'mse', 'mae']

    for run_id in pbar:
        random_seed = 42 + run_id

        # 对这次划分应用三种方法
        run_result = evaluate_one_split_with_methods(
            all_modalities=all_modalities,
            modality_combinations=modality_combinations,
            X_data=X_data,
            y_data=y_data,
            feature_counts_dict=feature_counts_dict,
            n_classes=n_classes,
            model_cache=model_cache,
            run_id=run_id,
            split_id=0,
            random_seed=random_seed
        )

        # 保存权重
        weights_file = os.path.join(output_dirs['ma_weights'], f"weights_run_{run_id:03d}.pkl")
        with open(weights_file, 'wb') as f:
            pickle.dump(run_result, f)

        # 保存预测结果
        for combo_name, combo_results in run_result.items():
            for method in ['CV', 'AIC', 'BIC', 'Equal']:  # 修改：增加 Equal
                if method in combo_results['predictions']:
                    pred_data = {
                        'probs': combo_results['predictions'][method]['probs'],
                        'labels': combo_results['predictions'][method]['labels'],
                        'true_labels': combo_results['true_labels']
                    }

                    pred_file = os.path.join(
                        output_dirs['predictions'],
                        f"pred_run_{run_id:03d}_{combo_name}_{method}.pkl"
                    )
                    with open(pred_file, 'wb') as f:
                        pickle.dump(pred_data, f)

        # 收集评估结果
        for combo_name, combo_results in run_result.items():
            if combo_name not in all_run_results:
                all_run_results[combo_name] = {
                    'CV': {metric: [] for metric in metrics_list},
                    'AIC': {metric: [] for metric in metrics_list},
                    'BIC': {metric: [] for metric in metrics_list},
                    'Equal': {metric: [] for metric in metrics_list}  # 新增 Equal
                }

            for method in ['CV', 'AIC', 'BIC', 'Equal']:  # 修改：增加 Equal
                if method in combo_results['predictions']:
                    metrics = combo_results['predictions'][method]['metrics']
                    for metric_name in metrics.keys():
                        if metric_name in all_run_results[combo_name][method]:
                            all_run_results[combo_name][method][metric_name].append(metrics[metric_name])

        # 更新进度条
        pbar.set_postfix({
            '当前运行': run_id + 1,
            '总运行': n_runs
        })

    elapsed_time = time.time() - start_time
    print(f"\n所有评估完成，总耗时: {elapsed_time:.2f} 秒")

    # 5. 保存评估结果
    print("\n5. 保存评估结果...")

    # 更新指标显示名称
    metric_displays = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'PR_AUC',
                       'Log_Loss', 'MSE', 'MAE']

    for method in ['CV', 'AIC', 'BIC', 'Equal']:  # 修改：增加 Equal
        # 为每种方法创建汇总表格
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
                        method in all_run_results[combo_name] and
                        metric_name in all_run_results[combo_name][method]):

                    values = all_run_results[combo_name][method][metric_name]
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
        csv_file = os.path.join(output_dirs['evaluations'], f"{method}_summary_100runs.csv")
        df_table.to_csv(csv_file, index=False)

        # 保存详细结果
        detailed_results = {}
        for combo_name in modality_combinations.keys():
            if combo_name in all_run_results and method in all_run_results[combo_name]:
                detailed_results[combo_name] = {}
                for metric_name in metrics_list:
                    if metric_name in all_run_results[combo_name][method]:
                        values = all_run_results[combo_name][method][metric_name]
                        if len(values) > 0:
                            detailed_results[combo_name][metric_name] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'max': float(np.max(values)),
                                'min': float(np.min(values)),
                                'values': [float(v) for v in values]
                            }

        detailed_file = os.path.join(output_dirs['evaluations'], f"{method}_detailed_results.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"  {method}方法结果已保存:")
        print(f"    - 汇总表格: {csv_file}")
        print(f"    - 详细结果: {detailed_file}")

    # 6. 创建综合比较表格
    print("\n6. 创建综合比较表格...")

    # 为每个指标创建比较表格
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

        comparison_data = []

        for combo_name in modality_combinations.keys():
            row = {'Model Combination': combo_name}

            for method in ['CV', 'AIC', 'BIC', 'Equal']:  # 修改：增加 Equal
                if (combo_name in all_run_results and
                        method in all_run_results[combo_name] and
                        metric_name in all_run_results[combo_name][method]):

                    values = all_run_results[combo_name][method][metric_name]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        row[f'{method}_Mean'] = f"{mean_val:.4f}"
                        row[f'{method}_Std'] = f"{std_val:.4f}"
                    else:
                        row[f'{method}_Mean'] = "N/A"
                        row[f'{method}_Std'] = "N/A"
                else:
                    row[f'{method}_Mean'] = "N/A"
                    row[f'{method}_Std'] = "N/A"

            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(output_dirs['summary'], f"{metric_display}_comparison.csv")
        df_comparison.to_csv(comparison_file, index=False)
        print(f"  {metric_display}比较表格已保存: {comparison_file}")

    # 7. 打印总结
    print("\n" + "=" * 80)
    print("评估完成总结")
    print("=" * 80)

    print(f"\n输出文件结构:")
    print(f"1. 权重文件目录: {output_dirs['ma_weights']}")
    print(f"   包含100个文件: weights_run_000.pkl 到 weights_run_099.pkl")
    print(f"   每个文件包含所有模态组合的四种权重 (CV, AIC, BIC, Equal)")

    print(f"\n2. 预测结果目录: {output_dirs['predictions']}")
    print(f"   包含400个文件: pred_run_000_*_CV.pkl, pred_run_000_*_AIC.pkl, "
          f"pred_run_000_*_BIC.pkl, pred_run_000_*_Equal.pkl 等")

    print(f"\n3. 评估结果目录: {output_dirs['evaluations']}")
    print(f"   包含8个文件:")
    print(f"   - CV_summary_100runs.csv")
    print(f"   - CV_detailed_results.json")
    print(f"   - AIC_summary_100runs.csv")
    print(f"   - AIC_detailed_results.json")
    print(f"   - BIC_summary_100runs.csv")
    print(f"   - BIC_detailed_results.json")
    print(f"   - Equal_summary_100runs.csv")
    print(f"   - Equal_detailed_results.json")

    print(f"\n4. 综合比较目录: {output_dirs['summary']}")
    print(f"   包含9个比较表格文件 (Accuracy, Precision, Recall, F1-Score, ROC_AUC, PR_AUC, Log_Loss, MSE, MAE)")

    print(f"\n实验配置:")
    print(f"- 随机划分次数: {n_runs}")
    print(f"- 权重选取方法: CV, AIC, BIC, Equal (新增平均权重作为基准)")
    print(f"- 评估指标: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, MSE, MAE (新增3个指标)")
    print(f"- 模态组合数量: {len(modality_combinations)}")
    print(f"- 总样本数: {len(y_data)}")
    print(f"- 类别数量: {n_classes}")
    print(f"- 总耗时: {elapsed_time:.2f} 秒")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()