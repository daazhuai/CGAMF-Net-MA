import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from itertools import combinations
import time
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')


def read_csv_file(path):
    """读取CSV文件"""
    return pd.read_csv(path)


# def normalize_sample_id(x):
#     """标准化样本ID"""
#     return str(x).split("-01")[0].split("-02")[0]
# metabric
def normalize_sample_id(x):
    return str(x)  # 或者根据需要处理


# # 文件路径
# clinical_path = "/root/BRCA/data/clinical_filtered.csv"
# snv_path = "/root/BRCA/data/SNV_filtered.csv"
# cnv_path = "/root/BRCA/data/CNV_filtered.csv"
# mrna_path = "/root/BRCA/data/mRNA_filtered.csv"

# metabric
clinical_path = "/root/METAtrain/data/metabric_clinical.csv"
snv_path = "/root/METAtrain/data/metabric_SNV.csv"
cnv_path = "/root/METAtrain/data/metabric_CNV.csv"
mrna_path = "/root/METAtrain/data/metabric_mRNA.csv"

# 读取数据
clin = read_csv_file(clinical_path)
snv = read_csv_file(snv_path)
cnv = read_csv_file(cnv_path)
mrna = read_csv_file(mrna_path)


def set_index_and_clean(df, is_clinical=False, return_encoders=False):
    """设置索引并清理数据，可选择返回编码器"""
    df = df.copy()

    # 1. 获取样本ID列并设置成索引
    sample_id_columns = ['SAMPLE_ID', 'Sample_ID', 'sample_id', 'sample']
    sample_col = None

    for col in sample_id_columns:
        if col in df.columns:
            sample_col = col
            break

    if sample_col is None and len(df.columns) > 0:
        sample_col = df.columns[0]
        print(f"警告: 未找到标准样本ID列名，使用第一列 '{sample_col}' 作为样本ID")

    if sample_col:
        df[sample_col] = df[sample_col].astype(str).map(normalize_sample_id)
        df = df.set_index(sample_col)
        df = df.reset_index()
        df = df.set_index(sample_col)

    # 2. 提取分类变量并编码
    clinical_encoders = {}  # 用于保存编码器

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

        print(f"分类变量列: {categorical_cols}")
        print(f"数值变量列: {numerical_cols}")

        for col in categorical_cols:
            try:
                le = LabelEncoder()
                # 先转换为字符串，并处理缺失值
                df[col] = df[col].astype(str).fillna('MISSING')
                df[col] = le.fit_transform(df[col])

                # 保存编码器及其映射关系
                clinical_encoders[col] = {
                    'encoder': le,
                    'classes': le.classes_.tolist(),
                    'mapping': {str(cls): int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
                }
                print(f"已编码列: {col} (唯一值数量: {len(le.classes_)})")
                print(f"  编码映射: {clinical_encoders[col]['mapping']}")

            except Exception as e:
                print(f"警告: 无法编码列 {col}: {e}")
                df = df.drop(columns=[col])

        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    if return_encoders:
        return df, clinical_encoders
    else:
        return df


# 处理各数据表
print("处理clinical数据...")
clin, clinical_encoders = set_index_and_clean(clin, is_clinical=True, return_encoders=True)

print("\n处理SNV数据...")
snv = set_index_and_clean(snv)

print("\n处理CNV数据...")
cnv = set_index_and_clean(cnv)

print("\n处理mRNA数据...")
mrna = set_index_and_clean(mrna)

print("\n数据形状:")
print(f"Clinical: {clin.shape}")
print(f"SNV: {snv.shape}")
print(f"CNV: {cnv.shape}")
print(f"mRNA: {mrna.shape}")


def clean_numeric_data(df, name):
    """清理数据框中的非数值数据"""
    df = df.copy()

    non_numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
        except:
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"警告: {name} 数据中存在非数值列，将被删除: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    return df


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

print(f"\n匹配样本数量: {len(common_samples)}")

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

print("\n原始特征维度:")
print(f"SNV: {X_snv_raw.shape}")
print(f"CNV: {X_cnv_raw.shape}")
print(f"mRNA: {X_mrna_raw.shape}")
print(f"Clinical: {X_clin_raw.shape}")

# 编码标签
le_y = LabelEncoder()
y_enc = le_y.fit_transform(y)
n_classes = len(np.unique(y_enc))

print(f"\n类别数量: {n_classes}")
print(f"类别标签: {le_y.classes_}")

# 标准化所有数据
print("\n标准化所有数据（完整数据集）...")
scaler_snv = StandardScaler()
scaler_cnv = StandardScaler()
scaler_mrna = StandardScaler()
scaler_clin = StandardScaler()

X_snv_std = scaler_snv.fit_transform(X_snv_raw)
X_cnv_std = scaler_cnv.fit_transform(X_cnv_raw)
X_mrna_std = scaler_mrna.fit_transform(X_mrna_raw)
X_clin_std = scaler_clin.fit_transform(X_clin_raw)

print("标准化完成")


# ==================================================================
# 定义神经网络组件
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

        # 根据dims中的键创建对应的MLP
        self.mlp = nn.ModuleDict({
            k: MLP(dims[k], hidden) for k in dims
        })

        # 为除clin外的所有模态创建门控
        self.gate = nn.ModuleDict({
            k: Gate(hidden) for k in dims if k != "clin"
        })

        self.classifier = nn.Linear(hidden, n_class)

        # 记录使用的模态
        self.used_modalities = list(dims.keys())

    def forward(self, xs):
        # 只使用模型定义的模态
        z = {k: self.mlp[k](xs[k]) for k in self.used_modalities}

        if "clin" in z:
            z_ref = z["clin"]
            fused = z_ref
            for k in z:
                if k != "clin":
                    fused = fused + self.gate[k](z[k], z_ref)
        else:
            # 如果没有clinical数据，使用第一个模态作为参考
            first_key = list(z.keys())[0]
            z_ref = z[first_key]
            fused = z_ref
            for k in z:
                if k != first_key:
                    fused = fused + self.gate[k](z[k], z_ref)

        return self.classifier(fused)


# ==================================================================
# 转换为PyTorch张量
X_dict = {
    "snv": torch.tensor(X_snv_std, dtype=torch.float32),
    "cnv": torch.tensor(X_cnv_std, dtype=torch.float32),
    "mrna": torch.tensor(X_mrna_std, dtype=torch.float32),
    "clin": torch.tensor(X_clin_std, dtype=torch.float32),
}
y_tensor = torch.tensor(y_enc, dtype=torch.long)

# ==================================================================
# 定义不同模态组合（增加Clinical单模态）
modal_combinations = {
    "clin": ["clin"],  # 新增：Clinical单模态，1个模型
    "clin_cnv": ["clin", "cnv"],  # 2个模态，3个候选模型
    "clin_snv": ["clin", "snv"],  # 2个模态，3个候选模型
    "clin_mrna": ["clin", "mrna"],  # 2个模态，3个候选模型
    "clin_cnv_snv": ["clin", "cnv", "snv"],  # 3个模态，7个候选模型
    "clin_cnv_mrna": ["clin", "cnv", "mrna"],  # 3个模态，7个候选模型
    "clin_snv_mrna": ["clin", "snv", "mrna"],  # 3个模态，7个候选模型
    "clin_cnv_snv_mrna": ["clin", "cnv", "snv", "mrna"],  # 4个模态，15个候选模型
}


# ==================================================================
# 训练单个模型的函数
def train_model(modalities_list, train_X_dict, train_y, val_X_dict=None, val_y=None,
                n_class=2, hidden_dim=128, epochs=50, lr=0.001):
    """训练单个模型"""
    # 创建模型维度字典
    dims = {}
    for mod in modalities_list:
        if mod == "snv":
            dims["snv"] = train_X_dict["snv"].shape[1] if isinstance(train_X_dict["snv"], torch.Tensor) else \
                train_X_dict["snv"].shape[1]
        elif mod == "cnv":
            dims["cnv"] = train_X_dict["cnv"].shape[1] if isinstance(train_X_dict["cnv"], torch.Tensor) else \
                train_X_dict["cnv"].shape[1]
        elif mod == "mrna":
            dims["mrna"] = train_X_dict["mrna"].shape[1] if isinstance(train_X_dict["mrna"], torch.Tensor) else \
                train_X_dict["mrna"].shape[1]
        elif mod == "clin":
            dims["clin"] = train_X_dict["clin"].shape[1] if isinstance(train_X_dict["clin"], torch.Tensor) else \
                train_X_dict["clin"].shape[1]

    model = MultiOmicNet(dims, hidden_dim, n_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 准备训练数据
    train_dataset = TensorDataset(*[train_X_dict[mod] for mod in modalities_list], train_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_data in train_loader:
            batch_X_dict = {}
            for i, mod in enumerate(modalities_list):
                batch_X_dict[mod] = batch_data[i].to(device)
            batch_y = batch_data[-1].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X_dict)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # 验证（如果提供了验证集）
    val_predictions = None
    val_labels = None
    if val_X_dict is not None and val_y is not None:
        model.eval()
        val_dataset = TensorDataset(*[val_X_dict[mod] for mod in modalities_list], val_y)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_probs = []
        all_true_labels = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_X_dict = {}
                for i, mod in enumerate(modalities_list):
                    batch_X_dict[mod] = batch_data[i].to(device)
                batch_y = batch_data[-1]

                outputs = model(batch_X_dict)
                probs = F.softmax(outputs, dim=1)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true_labels.extend(batch_y.numpy())

        val_predictions = np.array(all_probs)
        val_labels = np.array(all_true_labels)

    return model, val_predictions, val_labels


# ==================================================================
# 使用二次规划求解权重的函数
def solve_quadratic_program_scipy(Q_matrix):
    """使用SciPy求解二次规划问题"""
    from scipy.optimize import minimize

    n_models = Q_matrix.shape[0]

    def objective(w):
        return 0.5 * w @ Q_matrix @ w

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, 1) for _ in range(n_models)]

    initial_guess = np.ones(n_models) / n_models

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
    )

    weights = np.maximum(result.x, 0)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else initial_guess

    return weights


def solve_quadratic_program_cvxopt(Q_matrix):
    """使用cvxopt求解二次规划问题"""
    try:
        from cvxopt import matrix, solvers

        n_models = Q_matrix.shape[0]

        P = matrix(Q_matrix.astype(float))
        q = matrix(np.zeros(n_models))

        G = matrix(-np.eye(n_models))
        h = matrix(np.zeros(n_models))

        A = matrix(np.ones((1, n_models)).astype(float))
        b = matrix(1.0)

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
        print("cvxopt未安装，使用SciPy替代")
        return solve_quadratic_program_scipy(Q_matrix)


# ==================================================================
# 生成候选模型的辅助函数
def generate_candidate_models(available_modals):
    """生成候选模型：所有可能的非空子集"""
    candidate_models = []

    # 生成所有非空子集
    for r in range(1, len(available_modals) + 1):
        for combo in combinations(available_modals, r):
            candidate_models.append(list(combo))

    return candidate_models


# ==================================================================
# 主训练循环
print("\n" + "=" * 80)
print("开始训练不同模态组合的CV模型平均集成")
print("=" * 80)

K = 5  # 5折交叉验证
kf = KFold(n_splits=K, shuffle=True, random_state=42)
n_samples = len(y_enc)

all_results = {}

# 遍历每种模态组合
for combo_name, available_modals in modal_combinations.items():
    print(f"\n{'=' * 60}")
    print(f"处理模态组合: {combo_name}")
    print(f"可用模态: {available_modals}")
    print('=' * 60)

    # 生成当前模态组合的所有候选模型（所有非空子集）
    candidate_models = generate_candidate_models(available_modals)

    # 预期模型数量
    expected_counts = {
        "clin": 1,  # 2^1 - 1 = 1
        "clin_cnv": 3,  # 2^2 - 1 = 3
        "clin_snv": 3,  # 2^2 - 1 = 3
        "clin_mrna": 3,  # 2^2 - 1 = 3
        "clin_cnv_snv": 7,  # 2^3 - 1 = 7
        "clin_cnv_mrna": 7,  # 2^3 - 1 = 7
        "clin_snv_mrna": 7,  # 2^3 - 1 = 7
        "clin_cnv_snv_mrna": 15,  # 2^4 - 1 = 15
    }

    n_models = len(candidate_models)
    expected = expected_counts.get(combo_name, 0)

    print(f"生成 {n_models} 个候选模型:")
    for i, model_modalities in enumerate(candidate_models):
        print(f"  模型 {i + 1:2d}: {model_modalities}")

    if expected > 0 and n_models != expected:
        print(f"警告: 期望{expected}个模型，但生成了{n_models}个")

    # 如果是单模态，不需要进行二次规划优化
    if combo_name == "clin":
        print(f"\n单模态组合，直接训练单个模型...")

        # 创建保存目录
        save_dir = f"/root/METAtrain/new/saved_models_{combo_name}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存标准化器
        scalers = {
            'snv': scaler_snv,
            'cnv': scaler_cnv,
            'mrna': scaler_mrna,
            'clin': scaler_clin
        }

        with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)

        # 保存标签编码器
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(le_y, f)

        # 保存临床特征编码器
        with open(os.path.join(save_dir, 'clinical_encoders.pkl'), 'wb') as f:
            pickle.dump(clinical_encoders, f)
        print(f"  临床特征编码器已保存到: {os.path.join(save_dir, 'clinical_encoders.pkl')}")

        # 训练单个模型
        print(f"  训练模型 1/1: {candidate_models[0]}")

        # 在完整数据集上训练
        model, _, _ = train_model(
            candidate_models[0],
            X_dict,
            y_tensor,
            n_class=n_classes,
            hidden_dim=128,
            epochs=50,
            lr=0.001
        )

        # 保存模型
        model_name = f"model_1_clin.pth"
        model_path = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), model_path)

        # 在训练集上评估
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_inputs = {'clin': X_dict['clin'].to(device)}
        with torch.no_grad():
            outputs = model(test_inputs)
            probs = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

        train_accuracy = accuracy_score(y_enc, pred_labels)

        # 保存配置
        ensemble_config = {
            'combo_name': combo_name,
            'available_modals': available_modals,
            'candidate_models': candidate_models,
            'model_weights': [1.0],  # 单模型权重为1
            'n_classes': n_classes,
            'class_names': le_y.classes_.tolist(),
            'feature_dimensions': {
                'snv': X_snv_std.shape[1],
                'cnv': X_cnv_std.shape[1],
                'mrna': X_mrna_std.shape[1],
                'clin': X_clin_std.shape[1]
            },
            'clinical_encoding': {
                col: info['mapping'] for col, info in clinical_encoders.items()
            },
            'models_info': {
                'model_1': {
                    'modalities': candidate_models[0],
                    'weight': 1.0,
                    'filename': model_name,
                    'feature_dims': {'clin': X_clin_std.shape[1]}
                }
            },
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': n_samples,
            'n_models': 1
        }

        with open(os.path.join(save_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(ensemble_config, f, indent=4)

        # 保存结果
        all_results[combo_name] = {
            'save_dir': save_dir,
            'n_models': 1,
            'accuracy': train_accuracy,
            'weights': [1.0],
            'candidate_models': candidate_models
        }

        print(f"\n  Clinical单模态训练完成！")
        print(f"  训练集准确率: {train_accuracy:.4f}")
        print(f"  模型已保存到目录: {save_dir}")

        continue  # 跳过后续的多模态处理流程

    # ========== 以下是多模态组合的处理流程 ==========

    # 存储每个模型在每折的预测结果
    predictions = np.zeros((n_models, n_samples, n_classes))

    # K折交叉验证
    print(f"\n进行{K}折交叉验证...")
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f"  第 {fold_idx + 1}/{K} 折...")

        # 准备当前折的数据
        fold_train_X_dict = {k: X_dict[k][train_fold_idx] for k in X_dict}
        fold_train_y = y_tensor[train_fold_idx]
        fold_val_X_dict = {k: X_dict[k][val_fold_idx] for k in X_dict}
        fold_val_y = y_tensor[val_fold_idx]

        # 对每个候选模型进行训练和预测
        for model_idx, model_modalities in enumerate(candidate_models):
            if fold_idx == 0 and model_idx < 5:  # 只打印前5个模型
                print(f"    训练模型 {model_idx + 1:2d}/{n_models}: {model_modalities}")
            elif fold_idx == 0 and model_idx == 5:
                print(f"    ... (省略中间模型)")
            elif fold_idx == 0 and model_idx >= n_models - 5:  # 打印最后5个模型
                print(f"    训练模型 {model_idx + 1:2d}/{n_models}: {model_modalities}")

            # 训练模型
            model, val_preds, _ = train_model(
                model_modalities,
                fold_train_X_dict,
                fold_train_y,
                fold_val_X_dict,
                fold_val_y,
                n_class=n_classes,
                hidden_dim=128,
                epochs=30,
                lr=0.001
            )

            # 将预测结果存储到正确的位置
            predictions[model_idx, val_fold_idx, :] = val_preds

    print("  交叉验证完成！")

    # 优化权重 - 使用二次规划
    print(f"\n优化模型权重...")

    # 计算误差矩阵
    Y_true = y_enc
    Y_pred_labels = np.zeros((n_models, n_samples))
    for m in range(n_models):
        Y_pred_labels[m] = np.argmax(predictions[m], axis=1)

    errors = np.zeros((n_models, n_samples))
    for m in range(n_models):
        errors[m] = (Y_pred_labels[m] != Y_true).astype(float)

    # 计算Q矩阵
    Q = (errors @ errors.T) / n_samples

    print(f"  Q矩阵对角线（各模型单独预测的误差 - 前5个）:")
    for i in range(min(5, n_models)):
        print(f"    模型 {i + 1:2d} ({str(candidate_models[i]):30s}): {Q[i, i]:.6f}")

    # 求解权重
    best_w = solve_quadratic_program_cvxopt(Q)

    # 在完整数据集上训练所有模型并保存
    print(f"\n在完整数据集上训练并保存所有模型...")

    # 创建保存目录
    save_dir = f"saved_models_{combo_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存标准化器
    scalers = {
        'snv': scaler_snv,
        'cnv': scaler_cnv,
        'mrna': scaler_mrna,
        'clin': scaler_clin
    }

    with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    # 保存标签编码器
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le_y, f)

    # 保存临床特征编码器
    with open(os.path.join(save_dir, 'clinical_encoders.pkl'), 'wb') as f:
        pickle.dump(clinical_encoders, f)
    print(f"  临床特征编码器已保存到: {os.path.join(save_dir, 'clinical_encoders.pkl')}")

    # 训练并保存每个模型
    models_info = {}
    print(f"  训练并保存 {n_models} 个模型...")
    for model_idx, model_modalities in enumerate(candidate_models):
        if model_idx < 5 or model_idx >= n_models - 5:  # 只打印首尾模型
            print(f"    训练模型 {model_idx + 1:2d}/{n_models}: {model_modalities}")

        # 在完整数据集上训练模型
        model, _, _ = train_model(
            model_modalities,
            X_dict,
            y_tensor,
            n_class=n_classes,
            hidden_dim=128,
            epochs=50,
            lr=0.001
        )

        # 保存模型
        model_name = f"model_{model_idx + 1}_{'_'.join(model_modalities)}.pth"
        model_path = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), model_path)

        models_info[f"model_{model_idx + 1}"] = {
            'modalities': model_modalities,
            'weight': float(best_w[model_idx]),
            'filename': model_name,
            'feature_dims': {}
        }

        # 记录特征维度
        for mod in model_modalities:
            if mod == "snv":
                models_info[f"model_{model_idx + 1}"]['feature_dims']['snv'] = X_snv_std.shape[1]
            elif mod == "cnv":
                models_info[f"model_{model_idx + 1}"]['feature_dims']['cnv'] = X_cnv_std.shape[1]
            elif mod == "mrna":
                models_info[f"model_{model_idx + 1}"]['feature_dims']['mrna'] = X_mrna_std.shape[1]
            elif mod == "clin":
                models_info[f"model_{model_idx + 1}"]['feature_dims']['clin'] = X_clin_std.shape[1]

    # 保存集成配置
    ensemble_config = {
        'combo_name': combo_name,
        'available_modals': available_modals,
        'candidate_models': candidate_models,
        'model_weights': best_w.tolist(),
        'n_classes': n_classes,
        'class_names': le_y.classes_.tolist(),
        'feature_dimensions': {
            'snv': X_snv_std.shape[1],
            'cnv': X_cnv_std.shape[1],
            'mrna': X_mrna_std.shape[1],
            'clin': X_clin_std.shape[1]
        },
        'clinical_encoding': {
            col: info['mapping'] for col, info in clinical_encoders.items()
        },
        'models_info': models_info,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': n_samples,
        'n_models': n_models
    }

    with open(os.path.join(save_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=4)

    # 验证保存的模型（在训练集上的表现）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_and_predict_ensemble_local(test_X_dict, ensemble_dir):
        """加载集成模型并进行预测"""
        with open(os.path.join(ensemble_dir, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)

        all_predictions = []
        model_weights = config['model_weights']
        model_weights = np.array(model_weights) / np.sum(model_weights)

        for model_idx in range(len(config['candidate_models'])):
            modalities = config['candidate_models'][model_idx]
            model_name = f"model_{model_idx + 1}_{'_'.join(modalities)}.pth"
            model_path = os.path.join(ensemble_dir, model_name)

            # 获取模型维度
            dims = {}
            for mod in modalities:
                dims[mod] = test_X_dict[mod].shape[1]

            # 创建模型并加载权重
            model = MultiOmicNet(dims, 128, config['n_classes'])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # 准备测试数据
            test_inputs = {mod: test_X_dict[mod].to(device) for mod in modalities}

            # 预测
            with torch.no_grad():
                outputs = model(test_inputs)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())

        # 加权平均
        all_predictions = np.array(all_predictions)
        final_predictions = np.zeros_like(all_predictions[0])
        for i in range(len(all_predictions)):
            final_predictions += model_weights[i] * all_predictions[i]

        return final_predictions


    # 在训练集上进行验证
    print("  验证模型性能...")
    train_predictions = load_and_predict_ensemble_local(X_dict, save_dir)
    train_pred_labels = np.argmax(train_predictions, axis=1)
    train_accuracy = accuracy_score(y_enc, train_pred_labels)

    # 保存结果
    all_results[combo_name] = {
        'save_dir': save_dir,
        'n_models': n_models,
        'accuracy': train_accuracy,
        'weights': best_w.tolist(),
        'candidate_models': candidate_models
    }

    # 打印权重信息
    print(f"\n  优化后的模型权重 (权重 > 0.001):")
    nonzero_count = 0
    for i in range(n_models):
        if best_w[i] > 0.001:
            print(f"    模型 {i + 1:2d} ({str(candidate_models[i]):30s}): {best_w[i]:.6f}")
            nonzero_count += 1

    # 计算CV准则值
    uniform_weights = np.ones(n_models) / n_models
    uniform_cv = 0.5 * uniform_weights @ Q @ uniform_weights
    optimized_cv = 0.5 * best_w @ Q @ best_w

    print(f"\n  CV准则值:")
    print(f"    均匀权重 (1/M) CV值: {uniform_cv:.6f}")
    print(f"    优化权重 CV值: {optimized_cv:.6f}")
    if uniform_cv > 0:
        improvement = (uniform_cv - optimized_cv) / uniform_cv * 100
        print(f"    改进比例: {improvement:.2f}%")

    print(f"  训练集准确率: {train_accuracy:.4f}")
    print(f"  有显著权重的模型数量: {nonzero_count}/{n_models}")
    print(f"  所有模型已保存到目录: {save_dir}")

# ==================================================================
# 生成总结报告
print("\n" + "=" * 80)
print("训练总结")
print("=" * 80)

summary = f"""
不同模态组合的CV模型平均集成训练完成！
==================================================
训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
数据集信息:
  - 总样本数: {n_samples}
  - 类别数: {n_classes}
  - 类别标签: {list(le_y.classes_)}

各模态组合训练结果:
"""

for combo_name, result in all_results.items():
    summary += f"""
  {combo_name}:
    - 模型数量: {result['n_models']}
    - 训练集准确率: {result['accuracy']:.4f}
    - 保存目录: {result['save_dir']}
    - 有显著权重的模型: {sum(1 for w in result['weights'] if w > 0.001)}/{result['n_models']}
"""

summary += f"\n所有模型已保存完成！\n"

# 打印详细的候选模型信息
summary += "\n各组合候选模型详情:\n"
for combo_name, result in all_results.items():
    summary += f"\n{combo_name} ({len(result['candidate_models'])}个模型):\n"
    for i, modalities in enumerate(result['candidate_models']):
        weight = result['weights'][i]
        if weight > 0.001 or combo_name == "clin":  # 单模态显示所有
            summary += f"  模型 {i + 1:2d}: {modalities} (权重: {weight:.6f})\n"

print(summary)

# 保存总结到文件
with open('training_summary_all_combinations.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n训练总结已保存到: training_summary_all_combinations.txt")

# ==================================================================
# 创建外部测试集使用示例代码
print("\n" + "=" * 80)
print("创建外部测试集使用示例代码")
print("=" * 80)

example_code = '''"""
外部测试集使用示例代码（支持不同模态组合）
用于加载保存的模型平均集成模型并进行预测
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import os

# 神经网络定义（与训练时相同）
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

def load_ensemble_model(ensemble_dir):
    """加载保存的集成模型"""
    with open(os.path.join(ensemble_dir, 'ensemble_config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(ensemble_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    with open(os.path.join(ensemble_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # 加载临床特征编码器（如果存在）
    clinical_encoders_path = os.path.join(ensemble_dir, 'clinical_encoders.pkl')
    if os.path.exists(clinical_encoders_path):
        with open(clinical_encoders_path, 'rb') as f:
            clinical_encoders = pickle.load(f)
    else:
        clinical_encoders = None

    return config, scalers, label_encoder, clinical_encoders

def preprocess_test_data(test_data_dict, scalers):
    """
    预处理测试数据
    test_data_dict: 包含原始测试数据的字典，键为模态名
    scalers: 训练时保存的标准化器
    """
    def clean_numeric_data(df, name):
        """清理数据框中的非数值数据"""
        df = df.copy()
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        return df

    processed_data = {}

    for modal, data in test_data_dict.items():
        if modal not in scalers:
            print(f"警告: 没有找到模态 {modal} 的标准化器")
            continue

        # 预处理数据
        data_clean = clean_numeric_data(data, modal)

        # 标准化
        if hasattr(scalers[modal], 'transform'):
            data_std = scalers[modal].transform(data_clean.values.astype(np.float32))
        else:
            data_std = data_clean.values.astype(np.float32)

        processed_data[modal] = torch.tensor(data_std, dtype=torch.float32)

    return processed_data

def predict_with_ensemble(test_data_processed, ensemble_dir):
    """使用集成模型进行预测"""
    config, scalers, label_encoder, clinical_encoders = load_ensemble_model(ensemble_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_predictions = []
    available_weights = []
    available_model_indices = []

    # 找出所有可用的模型
    for model_idx in range(len(config['candidate_models'])):
        modalities = config['candidate_models'][model_idx]

        # 检查测试数据是否包含该模型所需的所有模态
        missing_modalities = [mod for mod in modalities if mod not in test_data_processed]
        if not missing_modalities:
            available_model_indices.append(model_idx)
            available_weights.append(config['model_weights'][model_idx])

    if not available_model_indices:
        raise ValueError("没有模型可以用于预测，请检查测试数据是否包含足够的模态")

    # 归一化权重
    available_weights = np.array(available_weights)
    available_weights = available_weights / np.sum(available_weights)

    # 对每个可用模型进行预测
    for idx, model_idx in enumerate(available_model_indices):
        modalities = config['candidate_models'][model_idx]
        model_name = f"model_{model_idx+1}_{'_'.join(modalities)}.pth"
        model_path = os.path.join(ensemble_dir, model_name)

        # 获取模型维度
        dims = {}
        for mod in modalities:
            dims[mod] = test_data_processed[mod].shape[1]

        # 创建模型并加载权重
        model = MultiOmicNet(dims, 128, config['n_classes'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 准备测试数据
        test_inputs = {mod: test_data_processed[mod].to(device) for mod in modalities}

        # 预测
        with torch.no_grad():
            outputs = model(test_inputs)
            probs = F.softmax(outputs, dim=1)
            all_predictions.append(probs.cpu().numpy())

    # 加权平均
    all_predictions = np.array(all_predictions)
    final_predictions = np.zeros_like(all_predictions[0])

    for i in range(len(all_predictions)):
        final_predictions += available_weights[i] * all_predictions[i]

    # 获取预测标签
    pred_labels = np.argmax(final_predictions, axis=1)
    pred_labels_decoded = label_encoder.inverse_transform(pred_labels)

    return final_predictions, pred_labels_decoded

# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("不同模态组合的集成模型预测示例")
    print("=" * 60)

    # 示例：使用clin+cnv+snv组合
    ensemble_dir = "saved_models_clin_cnv_snv"  # 选择要使用的模型组合

    # 加载集成模型配置
    config, scalers, label_encoder, clinical_encoders = load_ensemble_model(ensemble_dir)

    print(f"使用模型组合: {config['combo_name']}")
    print(f"可用模态: {config['available_modals']}")
    print(f"模型数量: {config['n_models']}")
    print(f"类别: {config['class_names']}")

    print("\n使用方法:")
    print("1. 准备测试数据字典，格式如: {'clin': clin_data, 'cnv': cnv_data, 'snv': snv_data}")
    print("2. 调用 preprocess_test_data() 预处理数据")
    print("3. 调用 predict_with_ensemble() 进行预测")

    # 注意：测试数据需要包含与训练时相同的特征列
'''

# 保存示例代码
with open('external_test_example_all_combinations.py', 'w', encoding='utf-8') as f:
    f.write(example_code)

print(f"\n外部测试集使用示例代码已保存到: external_test_example_all_combinations.py")
print("\n" + "=" * 80)
print("所有模态组合（包括Clinical单模态）的CV模型平均集成已训练并保存完成！")
print("=" * 80)