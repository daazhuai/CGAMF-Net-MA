import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, log_loss,
                             mean_squared_error, mean_absolute_error)
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================
# 1. 神经网络定义（与训练时相同）
# ============================================================
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


# ============================================================
# 2. 新增：PDI、RSQ、CCP指标计算函数
# ============================================================
def calculate_ccp(y_true, y_pred_probs):
    """
    计算CCP (Correct Classification Proportion) = Accuracy
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    return accuracy_score(y_true, y_pred)


def calculate_pdi(y_true, y_pred_probs, n_classes):
    """
    计算PDI (Probability Distribution Index)
    """
    n = len(y_true)
    pdi_sum = 0

    for current_class in range(n_classes):
        # 获取当前类别的样本索引
        class_indices = np.where(y_true == current_class)[0]
        if len(class_indices) == 0:
            continue

        # 获取其他类别的样本索引
        other_indices = [np.where(y_true == j)[0] for j in range(n_classes) if j != current_class]

        # 对于当前类别的每个样本
        for idx in class_indices:
            prob_current = y_pred_probs[idx, current_class]
            product_term = 1

            # 计算该概率大于其他每个类别中所有样本对当前类别概率的数量
            for other_idx_list in other_indices:
                if len(other_idx_list) > 0:
                    probs_other = y_pred_probs[other_idx_list, current_class]
                    n_greater = np.sum(prob_current > probs_other)
                    product_term *= n_greater

            pdi_sum += product_term

    # 计算分母
    class_sizes = [np.sum(y_true == i) for i in range(n_classes)]
    if any(sz == 0 for sz in class_sizes):
        return np.nan

    denominator = n_classes * np.prod(class_sizes)
    return pdi_sum / denominator


def calculate_rsq(y_true, y_pred_probs):
    """
    计算RSQ (Multiclass R-squared)
    """
    n = len(y_true)
    n_classes = y_pred_probs.shape[1]

    # 计算每个类别的先验概率
    class_props = []
    for i in range(n_classes):
        class_props.append(np.sum(y_true == i) / n)

    rsq_sum = 0
    for i in range(n_classes):
        # 计算第i类的方差
        var_p = np.var(y_pred_probs[:, i], ddof=1)
        # 计算分母：pi_i * (1-pi_i)
        denom = class_props[i] * (1 - class_props[i])

        if denom > 0:
            rsq_sum += var_p / denom

    # 计算最终的RSQ
    rsq_value = (rsq_sum / n_classes) * (n - 1) / n
    return rsq_value


# ============================================================
# 3. 加载模型和配置
# ============================================================
def load_ensemble_model(ensemble_dir):
    """加载保存的集成模型"""
    print(f"加载集成模型从: {ensemble_dir}")

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
        print(f"警告: 未找到临床特征编码器文件: {clinical_encoders_path}")

    return config, scalers, label_encoder, clinical_encoders


# ============================================================
# 4. 预处理外部验证数据（使用训练时的编码映射）
# ============================================================
def normalize_sample_id(x):
    """标准化样本ID - 保持不变"""
    return str(x)


def clean_numeric_data(df, name):
    """清理数据框中的非数值数据（与训练时相同）"""
    df = df.copy()

    non_numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
        except:
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"  警告: {name} 数据中存在非数值列，将被删除: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    return df


def encode_clinical_data_with_encoders(clin_df, clinical_encoders):
    """
    使用训练时保存的编码器编码临床数据
    """
    print("  使用训练时保存的编码器处理临床数据...")

    encoded_df = clin_df.copy()

    # 遍历每个临床特征，使用对应的编码器
    for col, encoder_info in clinical_encoders.items():
        if col in encoded_df.columns:
            print(f"    编码{col}列...")

            # 获取编码器
            if isinstance(encoder_info, dict) and 'encoder' in encoder_info:
                # 新格式：包含encoder对象
                encoder = encoder_info['encoder']
                # 处理缺失值
                encoded_df[col] = encoded_df[col].astype(str).fillna('MISSING')

                # 对于新出现的类别，映射到已知类别或使用默认值
                try:
                    encoded_df[col] = encoder.transform(encoded_df[col])
                except ValueError as e:
                    print(f"      警告: 编码{col}时遇到未知类别，将尝试映射...")
                    # 处理未知类别：如果出现未知类别，可以映射到最常见的类别
                    known_classes = set(encoder.classes_)

                    def map_to_known(x):
                        if x in known_classes:
                            return encoder.transform([x])[0]
                        else:
                            # 未知类别映射到最常见的类别
                            print(f"        未知类别 '{x}' 映射到 '{encoder.classes_[0]}'")
                            return encoder.transform([encoder.classes_[0]])[0]

                    encoded_df[col] = encoded_df[col].apply(map_to_known)

            elif isinstance(encoder_info, dict) and 'mapping' in encoder_info:
                # 旧格式：只有映射字典
                mapping = encoder_info['mapping']

                def apply_mapping(x):
                    x_str = str(x)
                    if x_str in mapping:
                        return mapping[x_str]
                    else:
                        print(f"        未知类别 '{x_str}'，使用默认值")
                        return list(mapping.values())[0]

                encoded_df[col] = encoded_df[col].astype(str).apply(apply_mapping)

    # 处理Age列（数值型，保持不变）
    if 'Age' in encoded_df.columns:
        print("    处理Age列...")
        encoded_df['Age'] = pd.to_numeric(encoded_df['Age'], errors='coerce').fillna(0)

    return encoded_df


def process_clinical_data_for_test(clin_df, label_encoder_y, clinical_encoders=None):
    """处理临床数据"""
    print("  处理临床数据...")

    # 确保有Sample_ID列
    if 'Sample_ID' not in clin_df.columns:
        sample_col = None
        for col in ['Sample_ID', 'SAMPLE_ID', 'sample_id', 'sample']:
            if col in clin_df.columns:
                sample_col = col
                break

        if sample_col:
            clin_df = clin_df.rename(columns={sample_col: 'Sample_ID'})
        else:
            clin_df = clin_df.rename(columns={clin_df.columns[0]: 'Sample_ID'})

    # 分离标签和特征
    if 'SUBTYPE' in clin_df.columns:
        y = clin_df['SUBTYPE']
        feature_columns = [col for col in clin_df.columns if col not in ['Sample_ID', 'SUBTYPE']]
        clin_features = clin_df[feature_columns]
    else:
        raise ValueError("Clinical数据中找不到'SUBTYPE'列")

    # 使用训练时保存的编码器编码临床特征
    if clinical_encoders is not None:
        clin_features_encoded = encode_clinical_data_with_encoders(clin_features, clinical_encoders)
    else:
        print("  警告: 未提供临床特征编码器，使用硬编码规则")
        # 回退到硬编码规则
        clin_features_encoded = clin_features  # 简化处理

    # 保存样本ID
    sample_ids = clin_df['Sample_ID'].values

    # 编码标签
    print("  编码标签...")
    try:
        y_encoded = label_encoder_y.transform(y)
    except Exception as e:
        print(f"    警告: 标签编码失败: {e}")
        print(f"    训练数据标签类别: {label_encoder_y.classes_}")
        print(f"    测试数据标签值示例: {y.unique()[:5]}")

        y_encoded = []
        for label in y:
            label_str = str(label).strip()
            if label_str in label_encoder_y.classes_:
                y_encoded.append(label_encoder_y.transform([label_str])[0])
            else:
                print(f"      发现新标签: '{label_str}'，编码为-1")
                y_encoded.append(-1)
        y_encoded = np.array(y_encoded)

    return clin_features_encoded, y, y_encoded, sample_ids


def preprocess_external_data(data_dir, label_encoder_y, clinical_encoders=None):
    """预处理外部验证数据 - 针对TCGA数据"""
    print(f"\n加载外部验证数据从: {data_dir}")

    # 加载数据
    clin_path = os.path.join(data_dir, "clinical_filtered.csv")
    snv_path = os.path.join(data_dir, "SNV_filtered.csv")
    cnv_path = os.path.join(data_dir, "CNV_filtered.csv")
    mrna_path = os.path.join(data_dir, "mRNA_filtered.csv")

    clin_df = pd.read_csv(clin_path)
    snv_df = pd.read_csv(snv_path)
    cnv_df = pd.read_csv(cnv_path)
    mrna_df = pd.read_csv(mrna_path)

    print(f"数据形状:")
    print(f"  Clinical: {clin_df.shape}")
    print(f"  SNV: {snv_df.shape}")
    print(f"  CNV: {cnv_df.shape}")
    print(f"  mRNA: {mrna_df.shape}")

    # 处理临床数据 - 获取样本ID
    clin_features, y_true_series, y_true_encoded, sample_ids_clin = process_clinical_data_for_test(
        clin_df, label_encoder_y, clinical_encoders
    )

    # 处理其他模态数据
    print("处理SNV数据...")
    snv_features = snv_df.drop(columns=['Sample_ID'])
    snv_features = clean_numeric_data(snv_features, "SNV")

    print("处理CNV数据...")
    cnv_features = cnv_df.drop(columns=['Sample_ID'])
    cnv_features = clean_numeric_data(cnv_features, "CNV")

    print("处理mRNA数据...")
    mrna_features = mrna_df.drop(columns=['Sample_ID'])
    mrna_features = clean_numeric_data(mrna_features, "mRNA")

    # 获取各数据集的样本ID
    sample_ids_snv = snv_df['Sample_ID'].values
    sample_ids_cnv = cnv_df['Sample_ID'].values
    sample_ids_mrna = mrna_df['Sample_ID'].values

    # 设置索引
    clin_features.index = sample_ids_clin
    snv_features.index = sample_ids_snv
    cnv_features.index = sample_ids_cnv
    mrna_features.index = sample_ids_mrna

    # 创建y_true的Series
    y_true = pd.Series(y_true_series.values, index=sample_ids_clin)

    # 查找共同样本
    common_samples = list(
        set(clin_features.index) &
        set(snv_features.index) &
        set(cnv_features.index) &
        set(mrna_features.index)
    )

    print(f"\n共同样本数量: {len(common_samples)}")

    if not common_samples:
        print("错误: 没有共同样本，请检查数据")
        return None

    # 筛选共同样本
    clin_features = clin_features.loc[common_samples]
    snv_features = snv_features.loc[common_samples]
    cnv_features = cnv_features.loc[common_samples]
    mrna_features = mrna_features.loc[common_samples]
    y_true = y_true.loc[common_samples]

    # 筛选y_true_encoded
    id_to_idx = {id: i for i, id in enumerate(sample_ids_clin)}
    valid_indices = [id_to_idx[id] for id in common_samples]
    y_true_encoded = y_true_encoded[valid_indices]

    # 打印临床特征编码后的值分布
    print("\n临床特征编码后值分布:")
    for col in ['ER', 'PR', 'HER2', 'LN', 'MENOPAUSEstage', 'Age']:
        if col in clin_features.columns:
            value_counts = clin_features[col].value_counts().sort_index()
            print(f"  {col}: {dict(value_counts)}")

    # 检查是否有未见过的标签
    unseen_labels = set(str(label).strip() for label in y_true) - set(label_encoder_y.classes_)
    if unseen_labels:
        print(f"\n警告: 测试数据中有训练时未见过的标签: {unseen_labels}")
        print(f"训练数据标签类别: {label_encoder_y.classes_}")

    return {
        'clin': clin_features,
        'snv': snv_features,
        'cnv': cnv_features,
        'mrna': mrna_features,
        'y_true': y_true,
        'y_true_encoded': y_true_encoded,
        'sample_ids': common_samples
    }


# ============================================================
# 5. 准备测试数据张量
# ============================================================
def prepare_test_tensors(test_data, scalers, config):
    """准备测试数据张量"""
    print("\n准备测试数据...")

    # 转换数据类型
    X_test_clin = test_data['clin'].values.astype(np.float32)
    X_test_snv = test_data['snv'].values.astype(np.float32)
    X_test_cnv = test_data['cnv'].values.astype(np.float32)
    X_test_mrna = test_data['mrna'].values.astype(np.float32)

    print(f"原始特征维度:")
    print(f"  Clinical: {X_test_clin.shape}")
    print(f"  SNV: {X_test_snv.shape}")
    print(f"  CNV: {X_test_cnv.shape}")
    print(f"  mRNA: {X_test_mrna.shape}")

    # 检查训练时的特征维度
    print(f"\n训练时特征维度:")
    for mod in ['clin', 'snv', 'cnv', 'mrna']:
        if mod in config['feature_dimensions']:
            train_dim = config['feature_dimensions'][mod]
            test_dim = eval(f'X_test_{mod}').shape[1]
            print(f"  {mod}: 训练时={train_dim}, 测试时={test_dim}")
            if train_dim != test_dim:
                print(f"    警告: {mod}特征维度不匹配!")
                if test_dim > train_dim:
                    print(f"      截断{mod}特征到{train_dim}维")
                    if mod == 'clin':
                        X_test_clin = X_test_clin[:, :train_dim]
                    elif mod == 'snv':
                        X_test_snv = X_test_snv[:, :train_dim]
                    elif mod == 'cnv':
                        X_test_cnv = X_test_cnv[:, :train_dim]
                    elif mod == 'mrna':
                        X_test_mrna = X_test_mrna[:, :train_dim]
                else:
                    print(f"      填充{mod}特征到{train_dim}维")
                    padding = np.zeros((X_test_clin.shape[0], train_dim - test_dim), dtype=np.float32)
                    if mod == 'clin':
                        X_test_clin = np.hstack([X_test_clin, padding])
                    elif mod == 'snv':
                        X_test_snv = np.hstack([X_test_snv, padding])
                    elif mod == 'cnv':
                        X_test_cnv = np.hstack([X_test_cnv, padding])
                    elif mod == 'mrna':
                        X_test_mrna = np.hstack([X_test_mrna, padding])

    # 标准化（使用训练时的标准化器）
    print("\n标准化测试数据...")
    try:
        test_data_dict = {}

        for mod in ['clin', 'snv', 'cnv', 'mrna']:
            if mod in scalers and hasattr(eval(f'X_test_{mod}'), 'shape'):
                X_test = eval(f'X_test_{mod}')
                if X_test.shape[0] > 0:
                    if X_test.shape[1] != scalers[mod].mean_.shape[0]:
                        print(f"  警告: {mod}维度不匹配，尝试调整...")
                        target_dim = scalers[mod].mean_.shape[0]
                        if X_test.shape[1] > target_dim:
                            X_test = X_test[:, :target_dim]
                        else:
                            padding = np.zeros((X_test.shape[0], target_dim - X_test.shape[1]), dtype=np.float32)
                            X_test = np.hstack([X_test, padding])

                    X_test_std = scalers[mod].transform(X_test)
                    test_data_dict[mod] = torch.tensor(X_test_std, dtype=torch.float32)
                    print(f"  已标准化 {mod} 数据 ({X_test_std.shape})")

        print("标准化成功完成!")
        return test_data_dict

    except Exception as e:
        print(f"标准化失败: {e}")
        return None


# ============================================================
# 6. 使用集成模型进行预测
# ============================================================
def predict_with_ensemble(test_data_dict, config, ensemble_dir):
    """使用集成模型进行预测"""
    if test_data_dict is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    all_predictions = []
    model_weights = config['model_weights']
    model_weights = np.array(model_weights)

    # 只考虑权重大于0.001的模型
    significant_models = model_weights > 0.001
    significant_indices = np.where(significant_models)[0]

    print(f"总模型数: {len(model_weights)}")
    print(f"有显著权重的模型数: {len(significant_indices)}")

    for model_idx in significant_indices:
        modalities = config['candidate_models'][model_idx]
        model_name = f"model_{model_idx + 1}_{'_'.join(modalities)}.pth"
        model_path = os.path.join(ensemble_dir, model_name)

        # 检查测试数据是否包含该模型所需的所有模态
        missing_modalities = [mod for mod in modalities if mod not in test_data_dict]
        if missing_modalities:
            print(f"  警告: 模型 {model_idx + 1} 需要模态 {missing_modalities}，但测试数据中缺失。跳过此模型。")
            continue

        # 获取模型维度
        dims = {}
        for mod in modalities:
            if mod in test_data_dict:
                dims[mod] = test_data_dict[mod].shape[1]
            else:
                print(f"  错误: 模态 {mod} 不在test_data_dict中")
                continue

        try:
            # 创建模型并加载权重
            model = MultiOmicNet(dims, 128, config['n_classes'])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # 准备测试数据
            test_inputs = {mod: test_data_dict[mod].to(device) for mod in modalities}

            # 预测
            with torch.no_grad():
                outputs = model(test_inputs)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append({
                    'probs': probs.cpu().numpy(),
                    'weight': model_weights[model_idx]
                })

            print(f"  已加载模型 {model_idx + 1}: {modalities} (权重: {model_weights[model_idx]:.4f})")

        except Exception as e:
            print(f"  错误: 无法加载模型 {model_idx + 1}: {e}")

    if not all_predictions:
        print("没有模型可以用于预测")
        return None

    # 加权平均
    print("\n进行加权平均集成...")
    final_predictions = np.zeros_like(all_predictions[0]['probs'])
    total_weight = 0

    for pred_info in all_predictions:
        weight = pred_info['weight']
        final_predictions += weight * pred_info['probs']
        total_weight += weight

    # 归一化
    if total_weight > 0:
        final_predictions = final_predictions / total_weight

    return final_predictions


# ============================================================
# 6.5 新增：使用Clinical单模态模型进行预测
# ============================================================
def predict_with_clinical_only(test_data_dict, config, ensemble_dir):
    """使用Clinical单模态模型进行预测（无需权重）"""
    if test_data_dict is None or 'clin' not in test_data_dict:
        print("临床数据不存在，无法进行Clinical单模态预测")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 查找Clinical-only模型 - 改进查找逻辑
    clinical_model_path = None
    print(f"在目录 {ensemble_dir} 中查找Clinical-only模型...")

    for filename in os.listdir(ensemble_dir):
        if filename.endswith('.pth'):
            print(f"  检查文件: {filename}")
            # 检查是否只包含clin模态（不包含其他模态）
            if '_clin' in filename and '_snv' not in filename and '_cnv' not in filename and '_mrna' not in filename:
                clinical_model_path = os.path.join(ensemble_dir, filename)
                print(f"  找到Clinical-only模型: {filename}")
                break

    if clinical_model_path is None:
        print("未找到Clinical-only模型文件")
        # 列出目录中的所有.pth文件
        pth_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.pth')]
        print(f"目录中的.pth文件: {pth_files}")
        return None

    # 准备模型维度
    dims = {'clin': test_data_dict['clin'].shape[1]}
    print(f"临床特征维度: {dims['clin']}")

    try:
        # 创建模型并加载权重
        model = MultiOmicNet(dims, 128, config['n_classes'])

        # 加载权重，处理可能的设备不匹配
        state_dict = torch.load(clinical_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 准备测试数据
        test_inputs = {'clin': test_data_dict['clin'].to(device)}

        # 预测
        with torch.no_grad():
            outputs = model(test_inputs)
            probs = F.softmax(outputs, dim=1)
            predictions = probs.cpu().numpy()

        print(f"Clinical-only模型预测完成，预测概率形状: {predictions.shape}")
        return predictions

    except Exception as e:
        print(f"错误: 无法加载Clinical-only模型: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 7. 计算多个评估指标（增强版）
# ============================================================
def calculate_metrics(y_true_encoded, y_pred_probs, label_encoder):
    """
    计算多个评估指标：
    - 加权PR AUC
    - 加权ROC AUC
    - Log Loss（交叉熵损失）
    - MSE（均方误差）
    - MAE（平均绝对误差）
    - 加权Accuracy
    - 加权Precision
    - 加权Recall
    - 加权F1-score
    - CCP (Correct Classification Proportion)
    - PDI (Probability Distribution Index)
    - RSQ (Multiclass R-squared)
    """
    if y_pred_probs is None:
        return None

    # 过滤掉未见过的标签（编码为-1的样本）
    valid_indices = y_true_encoded != -1
    if not np.any(valid_indices):
        print("警告: 没有有效的标签用于评估（所有标签都是未见过的）")
        return None

    y_true_valid = y_true_encoded[valid_indices]
    y_pred_probs_valid = y_pred_probs[valid_indices]
    y_pred_labels = np.argmax(y_pred_probs_valid, axis=1)

    valid_count = np.sum(valid_indices)
    total_count = len(y_true_encoded)
    print(f"有效样本数: {valid_count}/{total_count} ({valid_count / total_count * 100:.1f}%)")

    n_classes = len(label_encoder.classes_)

    # 初始化指标存储
    pr_auc_scores = []
    roc_auc_scores = []
    class_weights = []

    print(f"\n各类别评估详情:")
    print(f"{'类别':<10} {'样本数':<8} {'权重':<8} {'PR AUC':<8} {'ROC AUC':<8}")

    # 计算每个类别的权重（样本比例）和指标
    for i in range(n_classes):
        class_mask = (y_true_valid == i)
        class_samples = np.sum(class_mask)
        class_weight = class_samples / len(y_true_valid) if len(y_true_valid) > 0 else 0
        class_weights.append(class_weight)

        y_true_binary = class_mask.astype(int)

        # 计算PR AUC
        if class_samples > 0:
            if len(np.unique(y_true_binary)) > 1:  # 确保正负样本都有
                try:
                    pr_auc = average_precision_score(y_true_binary, y_pred_probs_valid[:, i])
                    pr_auc_scores.append(pr_auc)
                except Exception as e:
                    pr_auc_scores.append(0)
                    pr_auc = 0
            else:
                pr_auc_scores.append(0)
                pr_auc = 0
        else:
            pr_auc_scores.append(0)
            pr_auc = 0

        # 计算ROC AUC
        if class_samples > 0:
            if len(np.unique(y_true_binary)) > 1:  # 确保正负样本都有
                try:
                    roc_auc = roc_auc_score(y_true_binary, y_pred_probs_valid[:, i])
                    roc_auc_scores.append(roc_auc)
                except Exception as e:
                    roc_auc_scores.append(0)
                    roc_auc = 0
            else:
                roc_auc_scores.append(0)
                roc_auc = 0
        else:
            roc_auc_scores.append(0)
            roc_auc = 0

        # 打印结果
        print(f"{label_encoder.classes_[i]:<10} {class_samples:<8} {class_weight:.4f} {pr_auc:.4f} {roc_auc:.4f}")

    # 计算加权PR AUC
    if pr_auc_scores and len(pr_auc_scores) == len(class_weights) and sum(class_weights) > 0:
        weighted_pr_auc = np.sum(np.array(pr_auc_scores) * np.array(class_weights)) / np.sum(class_weights)
    else:
        weighted_pr_auc = np.mean(pr_auc_scores) if pr_auc_scores else np.nan

    # 计算加权ROC AUC
    if roc_auc_scores and len(roc_auc_scores) == len(class_weights) and sum(class_weights) > 0:
        weighted_roc_auc = np.sum(np.array(roc_auc_scores) * np.array(class_weights)) / np.sum(class_weights)
    else:
        weighted_roc_auc = np.mean(roc_auc_scores) if roc_auc_scores else np.nan

    # 计算Log Loss（交叉熵损失）
    try:
        # 将真实标签转换为one-hot编码
        y_true_onehot = np.zeros_like(y_pred_probs_valid)
        y_true_onehot[np.arange(len(y_true_valid)), y_true_valid] = 1

        # 添加小epsilon避免log(0)
        epsilon = 1e-15
        y_pred_probs_clipped = np.clip(y_pred_probs_valid, epsilon, 1 - epsilon)
        log_loss_value = -np.mean(np.sum(y_true_onehot * np.log(y_pred_probs_clipped), axis=1))
    except Exception as e:
        print(f"计算Log Loss时出错: {e}")
        log_loss_value = np.nan

    # 计算MSE（均方误差）- 使用预测概率和one-hot标签
    try:
        y_true_onehot = np.zeros_like(y_pred_probs_valid)
        y_true_onehot[np.arange(len(y_true_valid)), y_true_valid] = 1
        mse_value = mean_squared_error(y_true_onehot, y_pred_probs_valid)
    except Exception as e:
        print(f"计算MSE时出错: {e}")
        mse_value = np.nan

    # 计算MAE（平均绝对误差）
    try:
        y_true_onehot = np.zeros_like(y_pred_probs_valid)
        y_true_onehot[np.arange(len(y_true_valid)), y_true_valid] = 1
        mae_value = mean_absolute_error(y_true_onehot, y_pred_probs_valid)
    except Exception as e:
        print(f"计算MAE时出错: {e}")
        mae_value = np.nan

    # 计算加权Accuracy
    try:
        weighted_accuracy = accuracy_score(y_true_valid, y_pred_labels)
    except Exception as e:
        print(f"计算加权Accuracy时出错: {e}")
        weighted_accuracy = np.nan

    # 计算加权Precision
    try:
        weighted_precision = precision_score(y_true_valid, y_pred_labels, average='weighted', zero_division=0)
    except Exception as e:
        print(f"计算加权Precision时出错: {e}")
        weighted_precision = np.nan

    # 计算加权Recall
    try:
        weighted_recall = recall_score(y_true_valid, y_pred_labels, average='weighted', zero_division=0)
    except Exception as e:
        print(f"计算加权Recall时出错: {e}")
        weighted_recall = np.nan

    # 计算加权F1-score
    try:
        weighted_f1 = f1_score(y_true_valid, y_pred_labels, average='weighted', zero_division=0)
    except Exception as e:
        print(f"计算加权F1-score时出错: {e}")
        weighted_f1 = np.nan

    # 计算CCP (等同于Accuracy)
    try:
        ccp_value = calculate_ccp(y_true_valid, y_pred_probs_valid)
    except Exception as e:
        print(f"计算CCP时出错: {e}")
        ccp_value = np.nan

    # 计算PDI
    try:
        pdi_value = calculate_pdi(y_true_valid, y_pred_probs_valid, n_classes)
    except Exception as e:
        print(f"计算PDI时出错: {e}")
        pdi_value = np.nan

    # 计算RSQ
    try:
        rsq_value = calculate_rsq(y_true_valid, y_pred_probs_valid)
    except Exception as e:
        print(f"计算RSQ时出错: {e}")
        rsq_value = np.nan

    # 打印汇总指标
    print(f"\n评估指标汇总:")
    print(f"  加权PR AUC: {weighted_pr_auc:.4f}")
    print(f"  加权ROC AUC: {weighted_roc_auc:.4f}")
    print(f"  Log Loss: {log_loss_value:.4f}")
    print(f"  MSE: {mse_value:.4f}")
    print(f"  MAE: {mae_value:.4f}")
    print(f"  加权Accuracy: {weighted_accuracy:.4f}")
    print(f"  加权Precision: {weighted_precision:.4f}")
    print(f"  加权Recall: {weighted_recall:.4f}")
    print(f"  加权F1-score: {weighted_f1:.4f}")
    print(f"  CCP: {ccp_value:.4f}")
    print(f"  PDI: {pdi_value:.4f}")
    print(f"  RSQ: {rsq_value:.4f}")

    return {
        'weighted_pr_auc': weighted_pr_auc,
        'weighted_roc_auc': weighted_roc_auc,
        'log_loss': log_loss_value,
        'mse': mse_value,
        'mae': mae_value,
        'weighted_accuracy': weighted_accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'ccp': ccp_value,
        'pdi': pdi_value,
        'rsq': rsq_value,
        'y_pred_labels': y_pred_labels,  # 新增：返回预测标签
        'y_true_valid': y_true_valid,  # 新增：返回真实标签
        'valid_indices': valid_indices,  # 新增：返回有效样本索引
        'valid_indices_int': np.where(valid_indices)[0]  # 新增：整数索引
    }


# ============================================================
# 7.5 新增：保存预测结果和统计预测类别分布
# ============================================================
def save_prediction_results(y_true_encoded, y_pred_probs, metrics_result, label_encoder,
                            combo_name, ensemble_dir, results_dir, sample_ids=None):
    """
    保存预测结果和统计预测类别分布
    """
    print(f"\n保存{combo_name}的预测结果...")

    # 创建组合专属的结果目录
    combo_result_dir = os.path.join(results_dir, f"predictions_{os.path.basename(ensemble_dir)}")
    os.makedirs(combo_result_dir, exist_ok=True)

    # 获取有效样本的索引 - 使用整数索引
    valid_indices_bool = metrics_result['valid_indices']
    valid_indices_int = metrics_result['valid_indices_int']

    y_true_valid = metrics_result['y_true_valid']
    y_pred_labels = metrics_result['y_pred_labels']

    # 获取预测概率
    y_pred_probs_valid = y_pred_probs[valid_indices_bool]

    # 获取每个样本的最高概率和对应类别
    max_probs = np.max(y_pred_probs_valid, axis=1)

    # 准备结果DataFrame - 修复索引问题
    if sample_ids is not None:
        # 使用整数索引获取对应的sample_ids
        sample_ids_valid = [sample_ids[i] for i in valid_indices_int]
    else:
        sample_ids_valid = [f"sample_{i}" for i in range(len(y_true_valid))]

    results_df = pd.DataFrame({
        'sample_id': sample_ids_valid,
        'true_label_encoded': y_true_valid,
        'true_label': label_encoder.inverse_transform(y_true_valid),
        'pred_label_encoded': y_pred_labels,
        'pred_label': label_encoder.inverse_transform(y_pred_labels),
        'max_probability': max_probs,
        'correct': (y_true_valid == y_pred_labels).astype(int)
    })

    # 添加每个类别的预测概率
    for i, class_name in enumerate(label_encoder.classes_):
        results_df[f'prob_{class_name}'] = y_pred_probs_valid[:, i]

    # 保存详细预测结果
    predictions_csv_path = os.path.join(combo_result_dir, f"predictions_detailed.csv")
    results_df.to_csv(predictions_csv_path, index=False)
    print(f"  详细预测结果已保存到: {predictions_csv_path}")

    # ===== 统计预测类别分布 =====
    print(f"\n  {combo_name} 预测类别分布统计:")

    # 1. 总体预测分布
    pred_distribution = Counter(y_pred_labels)
    pred_distribution_df = pd.DataFrame([
        {
            'class_encoded': class_id,
            'class_name': label_encoder.inverse_transform([class_id])[0],
            'predicted_count': count,
            'predicted_percentage': count / len(y_pred_labels) * 100
        }
        for class_id, count in sorted(pred_distribution.items())
    ])

    print(f"  总体预测分布:")
    for _, row in pred_distribution_df.iterrows():
        print(f"    {row['class_name']}: {row['predicted_count']}样本 ({row['predicted_percentage']:.1f}%)")

    # 2. 按真实类别统计预测分布（混淆矩阵形式）
    confusion_stats = []
    for true_class in np.unique(y_true_valid):
        true_class_mask = (y_true_valid == true_class)
        true_class_samples = np.sum(true_class_mask)
        true_class_name = label_encoder.inverse_transform([true_class])[0]

        if true_class_samples > 0:
            pred_for_true = y_pred_labels[true_class_mask]
            pred_dist = Counter(pred_for_true)

            for pred_class, count in pred_dist.items():
                pred_class_name = label_encoder.inverse_transform([pred_class])[0]
                confusion_stats.append({
                    'true_class_encoded': true_class,
                    'true_class_name': true_class_name,
                    'pred_class_encoded': pred_class,
                    'pred_class_name': pred_class_name,
                    'count': count,
                    'percentage_of_true': count / true_class_samples * 100
                })

    confusion_df = pd.DataFrame(confusion_stats)

    # 3. 正确分类 vs 错误分类统计
    correct_count = np.sum(results_df['correct'])
    incorrect_count = len(results_df) - correct_count

    accuracy_stats = pd.DataFrame([
        {'category': 'correct', 'count': correct_count, 'percentage': correct_count / len(results_df) * 100},
        {'category': 'incorrect', 'count': incorrect_count, 'percentage': incorrect_count / len(results_df) * 100}
    ])

    # 4. 按类别统计准确率
    class_accuracy = []
    for class_id in np.unique(y_true_valid):
        class_mask = (y_true_valid == class_id)
        class_total = np.sum(class_mask)
        class_correct = np.sum((y_true_valid == y_pred_labels) & class_mask)
        class_name = label_encoder.inverse_transform([class_id])[0]

        class_accuracy.append({
            'class_encoded': class_id,
            'class_name': class_name,
            'total_samples': class_total,
            'correct_predictions': class_correct,
            'accuracy': class_correct / class_total if class_total > 0 else 0
        })

    class_accuracy_df = pd.DataFrame(class_accuracy)

    # 保存所有统计结果
    # 保存预测分布
    pred_dist_csv_path = os.path.join(combo_result_dir, f"prediction_distribution.csv")
    pred_distribution_df.to_csv(pred_dist_csv_path, index=False)
    print(f"  预测分布已保存到: {pred_dist_csv_path}")

    # 保存混淆统计
    confusion_csv_path = os.path.join(combo_result_dir, f"confusion_by_true_class.csv")
    confusion_df.to_csv(confusion_csv_path, index=False)
    print(f"  混淆统计已保存到: {confusion_csv_path}")

    # 保存类别准确率
    class_acc_csv_path = os.path.join(combo_result_dir, f"class_accuracy.csv")
    class_accuracy_df.to_csv(class_acc_csv_path, index=False)
    print(f"  类别准确率已保存到: {class_acc_csv_path}")

    # 保存准确率统计
    acc_stats_csv_path = os.path.join(combo_result_dir, f"accuracy_stats.csv")
    accuracy_stats.to_csv(acc_stats_csv_path, index=False)

    # 生成预测报告文本
    report_path = os.path.join(combo_result_dir, f"prediction_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"预测报告 - {combo_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(y_true_encoded)}\n")
        f.write(f"有效样本数: {len(y_true_valid)}\n\n")

        f.write("1. 总体预测分布:\n")
        f.write("-" * 40 + "\n")
        for _, row in pred_distribution_df.iterrows():
            f.write(f"  {row['class_name']}: {row['predicted_count']}样本 ({row['predicted_percentage']:.1f}%)\n")
        f.write("\n")

        f.write("2. 各类别准确率:\n")
        f.write("-" * 40 + "\n")
        for _, row in class_accuracy_df.iterrows():
            f.write(
                f"  {row['class_name']}: {row['accuracy'] * 100:.1f}% ({row['correct_predictions']}/{row['total_samples']})\n")
        f.write("\n")

        f.write("3. 正确/错误分类统计:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  正确分类: {correct_count}样本 ({correct_count / len(results_df) * 100:.1f}%)\n")
        f.write(f"  错误分类: {incorrect_count}样本 ({incorrect_count / len(results_df) * 100:.1f}%)\n")
        f.write("\n")

        f.write("4. 混淆矩阵（按真实类别）:\n")
        f.write("-" * 40 + "\n")
        for _, row in confusion_df.iterrows():
            f.write(
                f"  真实{row['true_class_name']} -> 预测{row['pred_class_name']}: {row['count']}样本 ({row['percentage_of_true']:.1f}%)\n")

    print(f"  预测报告已保存到: {report_path}")

    return {
        'predictions_df': results_df,
        'pred_distribution': pred_distribution_df,
        'confusion_df': confusion_df,
        'class_accuracy': class_accuracy_df,
        'accuracy_stats': accuracy_stats
    }


# ============================================================
# 8. 主函数：评估所有8个模态组合（包括Clinical单模态）
# ============================================================
def main():
    # 8个模态组合的目录（METABRIC训练好的模型）
    ensemble_dirs = [
        "saved_models_clin",  # Clinical单模态（新增）
        "saved_models_clin_cnv",
        "saved_models_clin_snv",
        "saved_models_clin_mrna",
        "saved_models_clin_cnv_snv",
        "saved_models_clin_cnv_mrna",
        "saved_models_clin_snv_mrna",
        "saved_models_clin_cnv_snv_mrna"
    ]

    # 外部验证数据目录（TCGA数据）
    external_data_dir = "/root/METAtrain/data/external_validation/"

    # 结果存储
    all_results = []

    print("=" * 80)
    print("评估8个不同模态组合的集成模型（TCGA外部验证）")
    print("=" * 80)

    # 加载第一个模型配置来获取标签编码器和临床编码器
    if not ensemble_dirs:
        print("没有找到模型目录")
        return

    # 找到第一个存在的模型目录
    first_existing_dir = None
    for dir_path in ensemble_dirs:
        if os.path.exists(dir_path):
            first_existing_dir = dir_path
            break

    if first_existing_dir is None:
        print("没有找到任何有效的模型目录")
        return

    first_config, first_scalers, label_encoder_y, clinical_encoders = load_ensemble_model(first_existing_dir)

    # 打印临床编码器信息，确认编码映射
    if clinical_encoders:
        print("\n临床特征编码映射:")
        for col, info in clinical_encoders.items():
            if isinstance(info, dict) and 'mapping' in info:
                print(f"  {col}: {info['mapping']}")

    # 预处理外部验证数据
    test_data = preprocess_external_data(external_data_dir, label_encoder_y, clinical_encoders)

    if test_data is None:
        print("数据预处理失败，退出程序")
        return

    y_true_encoded = test_data['y_true_encoded']
    sample_ids = test_data['sample_ids']

    print(f"\nTCGA外部验证数据已加载，样本数: {len(y_true_encoded)}")

    # 统计标签分布
    print("\n标签分布:")
    for i, class_name in enumerate(label_encoder_y.classes_):
        count = np.sum(y_true_encoded == i)
        if count > 0:
            print(f"  {class_name}: {count}个样本 ({count / len(y_true_encoded) * 100:.1f}%)")

    # 创建主结果目录
    main_results_dir = os.path.join(external_data_dir, "all_modality_combinations_results")
    os.makedirs(main_results_dir, exist_ok=True)

    # 逐个评估每个模态组合
    print("\n" + "=" * 80)
    print("开始评估各模态组合")
    print("=" * 80)

    for i, ensemble_dir in enumerate(ensemble_dirs):
        if not os.path.exists(ensemble_dir):
            print(f"\n模型目录不存在: {ensemble_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"评估模态组合 {i + 1}/{len(ensemble_dirs)}: {ensemble_dir}")
        print('=' * 60)

        try:
            # 1. 加载当前组合的模型
            config, scalers, _, _ = load_ensemble_model(ensemble_dir)

            # 2. 准备测试数据张量
            test_data_dict = prepare_test_tensors(test_data, scalers, config)
            if test_data_dict is None:
                print("数据准备失败，跳过此组合")
                continue

            # 3. 根据模态组合类型选择预测方法
            if ensemble_dir == "saved_models_clin":
                # Clinical单模态：使用单模型预测
                print("\n" + "-" * 40)
                print("使用Clinical单模态模型进行预测")
                print("-" * 40)
                y_pred_probs = predict_with_clinical_only(test_data_dict, config, ensemble_dir)
                if y_pred_probs is not None:
                    print(f"Clinical单模态预测完成，预测概率形状: {y_pred_probs.shape}")
            else:
                # 其他多模态组合：使用集成模型预测
                y_pred_probs = predict_with_ensemble(test_data_dict, config, ensemble_dir)

            if y_pred_probs is None:
                print("预测失败，跳过此组合")
                continue

            # 4. 计算多个评估指标（同时获取预测标签）
            metrics = calculate_metrics(y_true_encoded, y_pred_probs, label_encoder_y)

            if metrics is not None:
                # 获取组合信息
                combo_name = config.get('combo_name', 'unknown')
                available_modals = config.get('available_modals', [])
                n_models = config.get('n_models', 0)

                # 存储结果
                result = {
                    'combo_id': i + 1,
                    'combo_name': combo_name,
                    'ensemble_dir': ensemble_dir,
                    'available_modals': available_modals,
                    'n_models': n_models,
                    'weighted_pr_auc': float(metrics['weighted_pr_auc']),
                    'weighted_roc_auc': float(metrics['weighted_roc_auc']),
                    'log_loss': float(metrics['log_loss']),
                    'mse': float(metrics['mse']),
                    'mae': float(metrics['mae']),
                    'weighted_accuracy': float(metrics['weighted_accuracy']),
                    'weighted_precision': float(metrics['weighted_precision']),
                    'weighted_recall': float(metrics['weighted_recall']),
                    'weighted_f1': float(metrics['weighted_f1']),
                    'ccp': float(metrics['ccp']),
                    'pdi': float(metrics['pdi']),
                    'rsq': float(metrics['rsq'])
                }
                all_results.append(result)

                print(f"\n评估完成: {combo_name}")
                print(f"  加权PR AUC = {metrics['weighted_pr_auc']:.4f}")
                print(f"  加权ROC AUC = {metrics['weighted_roc_auc']:.4f}")
                print(f"  Log Loss = {metrics['log_loss']:.4f}")
                print(f"  MSE = {metrics['mse']:.4f}")
                print(f"  MAE = {metrics['mae']:.4f}")
                print(f"  加权Accuracy = {metrics['weighted_accuracy']:.4f}")
                print(f"  加权Precision = {metrics['weighted_precision']:.4f}")
                print(f"  加权Recall = {metrics['weighted_recall']:.4f}")
                print(f"  加权F1-score = {metrics['weighted_f1']:.4f}")
                print(f"  CCP = {metrics['ccp']:.4f}")
                print(f"  PDI = {metrics['pdi']:.4f}")
                print(f"  RSQ = {metrics['rsq']:.4f}")

                # 5. 保存预测结果和统计预测类别分布
                save_prediction_results(
                    y_true_encoded=y_true_encoded,
                    y_pred_probs=y_pred_probs,
                    metrics_result=metrics,
                    label_encoder=label_encoder_y,
                    combo_name=combo_name,
                    ensemble_dir=ensemble_dir,
                    results_dir=main_results_dir,
                    sample_ids=sample_ids
                )

            else:
                print("无法计算评估指标")

        except Exception as e:
            print(f"评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存所有结果
    print("\n" + "=" * 80)
    print("保存评估结果")
    print("=" * 80)

    if all_results:
        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)

        # 保存结果到CSV（包含所有指标）
        csv_path = os.path.join(main_results_dir, "modality_combinations_all_metrics.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"结果已保存到: {csv_path}")

        # 保存结果到JSON
        json_path = os.path.join(main_results_dir, "modality_combinations_all_metrics.json")
        results_dict = results_df.to_dict('records')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"结果已保存到: {json_path}")

        # 打印结果汇总（按加权PR AUC排序）
        print("\n" + "=" * 80)
        print("评估结果汇总（按加权PR AUC排序）")
        print("=" * 80)

        results_df_sorted = results_df.sort_values('weighted_pr_auc', ascending=False)

        print(
            f"\n{'排名':<4} {'组合名称':<20} {'可用模态':<25} {'模型数':<6} {'PR AUC':<8} {'ROC AUC':<8} {'F1':<8} {'CCP':<8}")
        print("-" * 95)

        for idx, row in results_df_sorted.iterrows():
            rank = idx + 1
            combo_name = row['combo_name'][:18] + '...' if len(row['combo_name']) > 18 else row['combo_name']
            available_modals = ', '.join(row['available_modals'])[:23] + '...' if len(
                ', '.join(row['available_modals'])) > 23 else ', '.join(row['available_modals'])
            n_models = row['n_models']
            pr_auc = row['weighted_pr_auc']
            roc_auc = row['weighted_roc_auc']
            f1 = row['weighted_f1']
            ccp = row['ccp']

            print(
                f"{rank:<4} {combo_name:<20} {available_modals:<25} {n_models:<6} {pr_auc:<8.4f} {roc_auc:<8.4f} {f1:<8.4f} {ccp:<8.4f}")

        # 保存汇总报告
        report_path = os.path.join(main_results_dir, "evaluation_summary.txt")
        with open(report_path, 'w') as f:
            f.write("8个模态组合外部验证评估报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"评估时间: {pd.Timestamp.now()}\n")
            f.write(f"外部验证数据目录: {external_data_dir}\n")
            f.write(f"总样本数: {len(y_true_encoded)}\n")
            f.write(f"有效样本数: {len(y_true_encoded[y_true_encoded != -1])}\n\n")

            f.write("标签分布:\n")
            for i, class_name in enumerate(label_encoder_y.classes_):
                count = np.sum(y_true_encoded == i)
                if count > 0:
                    f.write(f"  {class_name}: {count}个样本 ({count / len(y_true_encoded) * 100:.1f}%)\n")
            f.write("\n")

            f.write("临床特征编码值分布:\n")
            for col in ['ER', 'PR', 'HER2', 'LN', 'MENOPAUSEstage']:
                if col in test_data['clin'].columns:
                    value_counts = test_data['clin'][col].value_counts().sort_index()
                    f.write(f"  {col}: {dict(value_counts)}\n")
            f.write("\n")

            f.write("评估结果汇总（按加权PR AUC排序）:\n")
            f.write("-" * 120 + "\n")
            f.write(
                f"{'排名':<4} {'组合名称':<20} {'可用模态':<25} {'模型数':<6} {'PR AUC':<8} {'ROC AUC':<8} {'Log Loss':<8} {'MSE':<8} {'MAE':<8} {'Acc':<8} {'F1':<8} {'CCP':<8} {'PDI':<8} {'RSQ':<8}\n")
            f.write("-" * 120 + "\n")

            for idx, row in results_df_sorted.iterrows():
                rank = idx + 1
                combo_name = row['combo_name']
                available_modals = ', '.join(row['available_modals'])
                n_models = row['n_models']
                pr_auc = row['weighted_pr_auc']
                roc_auc = row['weighted_roc_auc']
                log_loss = row['log_loss']
                mse = row['mse']
                mae = row['mae']
                acc = row['weighted_accuracy']
                f1 = row['weighted_f1']
                ccp = row['ccp']
                pdi = row['pdi']
                rsq = row['rsq']

                f.write(
                    f"{rank:<4} {combo_name:<20} {available_modals:<25} {n_models:<6} {pr_auc:<8.4f} {roc_auc:<8.4f} {log_loss:<8.4f} {mse:<8.4f} {mae:<8.4f} {acc:<8.4f} {f1:<8.4f} {ccp:<8.4f} {pdi:<8.4f} {rsq:<8.4f}\n")

            f.write("\n最佳模型组合（按加权PR AUC）:\n")
            best_result = results_df_sorted.iloc[0]
            f.write(f"  组合名称: {best_result['combo_name']}\n")
            f.write(f"  可用模态: {', '.join(best_result['available_modals'])}\n")
            f.write(f"  模型数量: {best_result['n_models']}\n")
            f.write(f"  加权PR AUC: {best_result['weighted_pr_auc']:.4f}\n")
            f.write(f"  加权ROC AUC: {best_result['weighted_roc_auc']:.4f}\n")
            f.write(f"  加权F1-score: {best_result['weighted_f1']:.4f}\n")
            f.write(f"  CCP: {best_result['ccp']:.4f}\n")
            f.write(f"  PDI: {best_result['pdi']:.4f}\n")
            f.write(f"  RSQ: {best_result['rsq']:.4f}\n")

        print(f"\n汇总报告已保存到: {report_path}")
        print(f"\n各模态组合的详细预测结果保存在: {main_results_dir}/predictions_* 目录下")

        # 打印Clinical单模态的结果
        clin_result = results_df[results_df['ensemble_dir'] == 'saved_models_clin']
        if not clin_result.empty:
            print("\n" + "=" * 60)
            print("Clinical单模态评估结果:")
            print("=" * 60)
            print(f"  组合名称: {clin_result.iloc[0]['combo_name']}")
            print(f"  可用模态: {clin_result.iloc[0]['available_modals']}")
            print(f"  加权PR AUC: {clin_result.iloc[0]['weighted_pr_auc']:.4f}")
            print(f"  加权ROC AUC: {clin_result.iloc[0]['weighted_roc_auc']:.4f}")
            print(f"  加权Accuracy: {clin_result.iloc[0]['weighted_accuracy']:.4f}")
            print(f"  加权F1-score: {clin_result.iloc[0]['weighted_f1']:.4f}")
            print(f"  CCP: {clin_result.iloc[0]['ccp']:.4f}")
        else:
            print("\n" + "=" * 60)
            print("警告: Clinical单模态评估结果未生成!")
            print("请检查:")
            print("1. saved_models_clin 目录是否存在")
            print("2. 目录中是否包含Clinical-only模型文件（如 model_1_clin.pth）")
            print("3. 模型文件格式是否正确")
            print("=" * 60)

    else:
        print("没有成功评估任何模态组合")

    print("\n" + "=" * 80)
    print("所有模态组合评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    # 设置OMP_NUM_THREADS环境变量
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    main()