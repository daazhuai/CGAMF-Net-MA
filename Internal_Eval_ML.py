import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    f1_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
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


# ======================== 机器学习模型定义 ========================
def create_ml_models():
    """创建四种机器学习模型"""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial' if n_classes > 2 else 'auto',
            solver='lbfgs' if n_classes > 2 else 'liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }
    return models


# ======================== 评估函数（新增指标） ========================
def evaluate_predictions(probs, preds, true_labels, n_classes):
    """评估预测结果（包含新增指标）"""
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


# ======================== 特征拼接函数 ========================
def concatenate_features(X_data, modalities):
    """根据模态组合拼接特征"""
    features_list = []
    for mod in modalities:
        features_list.append(X_data[mod])
    return np.hstack(features_list)


# ======================== 单次划分评估函数 ========================
def evaluate_ml_for_split(all_modalities, modality_combinations, X_data, y_data,
                          n_classes, run_id, random_seed=42):
    """
    对一次划分评估所有机器学习模型
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

    # 2. 准备数据字典
    train_X_dict = {}
    val_X_dict = {}

    for mod in all_modalities:
        X_mod = X_data[mod]

        scaler = StandardScaler()
        X_train = X_mod[train_idx]
        scaler.fit(X_train)

        X_train_std = scaler.transform(X_train)
        X_val_std = scaler.transform(X_mod[val_idx])

        train_X_dict[mod] = X_train_std
        val_X_dict[mod] = X_val_std

    val_y_np = y_data[val_idx]

    # 4. 评估每种模态组合的所有模型
    results = {}

    for combo_name, modalities in modality_combinations.items():
        # 拼接特征
        X_train_concat = concatenate_features(train_X_dict, modalities)
        X_val_concat = concatenate_features(val_X_dict, modalities)

        combo_results = {}

        # 对每个模态组合创建新的模型实例
        models = create_ml_models()  # 每次创建新模型，避免特征维度冲突

        for model_name, model in models.items():
            try:
                # 训练模型
                model.fit(X_train_concat, y_data[train_idx])

                # 预测
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_val_concat)
                else:
                    # 如果模型没有predict_proba，使用决策函数
                    if hasattr(model, 'decision_function'):
                        decision = model.decision_function(X_val_concat)
                        if n_classes == 2:
                            probs = np.vstack([1 - decision, decision]).T
                        else:
                            probs = decision
                    else:
                        preds = model.predict(X_val_concat)
                        probs = label_binarize(preds, classes=range(n_classes))

                # 确保概率矩阵形状正确
                if probs.shape[1] != n_classes and n_classes == 2:
                    if probs.shape[1] == 1:
                        probs = np.hstack([1 - probs, probs])

                pred_labels = np.argmax(probs, axis=1)

                # 评估（使用新增指标的评估函数）
                eval_metrics = evaluate_predictions(probs, pred_labels, val_y_np, n_classes)

                combo_results[model_name] = {
                    'probs': probs,
                    'labels': pred_labels,
                    'metrics': eval_metrics,
                    'true_labels': val_y_np.tolist()
                }

            except Exception as e:
                print(f"警告: {model_name} 在 {combo_name} 上训练失败: {str(e)}")
                combo_results[model_name] = None

        results[combo_name] = combo_results

    return results


# ======================== 数据预处理函数 ========================
# tcga
    # def normalize_sample_id(x):
    #     return str(x).split("-01")[0].split("-02")[0]

# metabric
def normalize_sample_id(x):
    return str(x)  # 或者根据实际格式处理


def read_csv_file(path):
    return pd.read_csv(path)


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


# ======================== 主程序 ========================
def main():
    print("=" * 80)
    print("机器学习模型评估 (4种模型 × 8种模态组合) - 包含Log Loss、MSE、MAE指标")
    print("=" * 80)

    # 设置随机种子
    set_seed(42)

    # 0. 数据准备
    print("\n0. 加载数据...")

    # 文件路径（请根据实际情况修改）
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
    global n_classes
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

    # 定义4种机器学习模型
    model_names = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']

    print(f"所有可用模态: {all_modalities}")
    print(f"将评估 {len(modality_combinations)} 种模态组合 × {len(model_names)} 种模型")
    for name, combo in modality_combinations.items():
        print(f"  {name}: {combo}")

    # 2. 创建输出目录
    print("\n2. 创建输出目录...")

    output_dirs = {
        'predictions': './ml_predictions',
        'evaluations': './ml_evaluations',
        'summary': './ml_summary'
    }

    for dir_name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_path}")

    # 3. 进行100次划分评估
    print("\n3. 开始100次划分评估...")
    start_time = time.time()

    n_runs = 100
    # 存储所有结果的结构：modality -> model -> metric -> list of values
    all_run_results = {}

    # 更新指标列表，包含新增的评估指标
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'prauc', 'log_loss', 'mse', 'mae']

    # 初始化结果数据结构
    for combo_name in modality_combinations.keys():
        all_run_results[combo_name] = {}
        for model_name in model_names:
            all_run_results[combo_name][model_name] = {metric: [] for metric in metrics_list}

    # 进度条
    pbar = tqdm(range(n_runs), desc="评估进度")

    for run_id in pbar:
        random_seed = 42 + run_id

        # 对这次划分评估所有模型
        run_result = evaluate_ml_for_split(
            all_modalities=all_modalities,
            modality_combinations=modality_combinations,
            X_data=X_data,
            y_data=y_data,
            n_classes=n_classes,
            run_id=run_id,
            random_seed=random_seed
        )

        # 保存预测结果
        for combo_name, model_results in run_result.items():
            for model_name, single_result in model_results.items():
                if single_result is not None:
                    pred_data = {
                        'probs': single_result['probs'],
                        'labels': single_result['labels'],
                        'true_labels': single_result['true_labels'],
                        'metrics': single_result['metrics']  # 现在包含9个指标
                    }

                    pred_file = os.path.join(
                        output_dirs['predictions'],
                        f"ml_pred_run_{run_id:03d}_{combo_name}_{model_name}.pkl"
                    )
                    with open(pred_file, 'wb') as f:
                        pickle.dump(pred_data, f)

        # 收集评估结果
        for combo_name, model_results in run_result.items():
            for model_name, single_result in model_results.items():
                if single_result is not None:
                    metrics = single_result['metrics']
                    for metric_name in metrics.keys():
                        if metric_name in all_run_results[combo_name][model_name]:
                            all_run_results[combo_name][model_name][metric_name].append(metrics[metric_name])

        # 更新进度条
        pbar.set_postfix({
            '当前运行': run_id + 1,
            '总运行': n_runs
        })

    elapsed_time = time.time() - start_time
    print(f"\n所有评估完成，总耗时: {elapsed_time:.2f} 秒")

    # 4. 保存评估结果
    print("\n4. 保存评估结果...")

    # 更新指标显示名称，包含新增指标
    metric_displays = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'PR_AUC',
                       'Log_Loss', 'MSE', 'MAE']

    # 为每种模型创建汇总表格
    for model_name in model_names:
        print(f"\n  处理 {model_name} 的结果...")

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
                        model_name in all_run_results[combo_name] and
                        metric_name in all_run_results[combo_name][model_name]):

                    values = all_run_results[combo_name][model_name][metric_name]
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
        csv_file = os.path.join(output_dirs['evaluations'], f"ml_{model_name}_summary_100runs.csv")
        df_table.to_csv(csv_file, index=False)
        print(f"    已保存: {csv_file}")

    # 保存详细结果
    detailed_results = {}
    for combo_name in modality_combinations.keys():
        detailed_results[combo_name] = {}
        for model_name in model_names:
            if model_name in all_run_results[combo_name]:
                detailed_results[combo_name][model_name] = {}
                for metric_name in metrics_list:
                    if metric_name in all_run_results[combo_name][model_name]:
                        values = all_run_results[combo_name][model_name][metric_name]
                        if len(values) > 0:
                            detailed_results[combo_name][model_name][metric_name] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'max': float(np.max(values)),
                                'min': float(np.min(values)),
                                'values': [float(v) for v in values]
                            }

    detailed_file = os.path.join(output_dirs['evaluations'], "ml_all_models_detailed_results.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n详细结果已保存: {detailed_file}")

    # 5. 创建最佳模型比较表格
    print("\n5. 创建最佳模型比较表格...")

    # 找出每种模态组合的最佳模型（基于平均F1分数）
    best_comparison = []

    for combo_name in modality_combinations.keys():
        best_f1 = 0
        best_model = ""

        for model_name in model_names:
            if (combo_name in all_run_results and
                    model_name in all_run_results[combo_name] and
                    'f1' in all_run_results[combo_name][model_name]):

                values = all_run_results[combo_name][model_name]['f1']
                if len(values) > 0:
                    mean_f1 = np.mean(values)
                    if mean_f1 > best_f1:
                        best_f1 = mean_f1
                        best_model = model_name

        if best_model:
            # 收集所有指标
            row = {'Modality Combination': combo_name, 'Best Model': best_model}

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

                if (combo_name in all_run_results and
                        best_model in all_run_results[combo_name] and
                        metric_name in all_run_results[combo_name][best_model]):

                    values = all_run_results[combo_name][best_model][metric_name]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        row[metric_display] = f"{mean_val:.4f}±{std_val:.4f}"
                    else:
                        row[metric_display] = "N/A"
                else:
                    row[metric_display] = "N/A"

            best_comparison.append(row)

    df_best = pd.DataFrame(best_comparison)
    best_file = os.path.join(output_dirs['summary'], "ml_best_models_comparison.csv")
    df_best.to_csv(best_file, index=False)
    print(f"  最佳模型比较: {best_file}")

    # 6. 创建跨模型比较表格（每种模态组合下所有模型的平均表现）
    print("\n6. 创建跨模型比较表格...")

    cross_model_comparison = []

    for combo_name in modality_combinations.keys():
        for model_name in model_names:
            if (combo_name in all_run_results and
                    model_name in all_run_results[combo_name]):

                row = {'Modality Combination': combo_name, 'Model': model_name}

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

                    if metric_name in all_run_results[combo_name][model_name]:
                        values = all_run_results[combo_name][model_name][metric_name]
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            row[metric_display] = f"{mean_val:.4f}±{std_val:.4f}"
                        else:
                            row[metric_display] = "N/A"
                    else:
                        row[metric_display] = "N/A"

                cross_model_comparison.append(row)

    df_cross = pd.DataFrame(cross_model_comparison)
    cross_file = os.path.join(output_dirs['summary'], "ml_cross_model_comparison.csv")
    df_cross.to_csv(cross_file, index=False)
    print(f"  跨模型比较: {cross_file}")

    # 7. 打印总结
    print("\n" + "=" * 80)
    print("评估完成总结")
    print("=" * 80)

    print(f"\n输出文件结构:")
    print(f"1. 预测结果目录: {output_dirs['predictions']}")
    print(
        f"   包含 {n_runs} × {len(modality_combinations)} × {len(model_names)} = {n_runs * len(modality_combinations) * len(model_names)} 个文件")
    print(f"   每个文件包含预测概率、预测标签、真实标签和9个评估指标")

    print(f"\n2. 评估结果目录: {output_dirs['evaluations']}")
    print(f"   包含:")
    print(f"   - 4个模型汇总表格 (ml_*_summary_100runs.csv) - 每个表格包含9个指标")
    print(f"   - 1个详细结果文件 (ml_all_models_detailed_results.json) - 包含9个指标的详细统计")

    print(f"\n3. 综合比较目录: {output_dirs['summary']}")
    print(f"   包含2个比较文件:")
    print(f"   - ml_best_models_comparison.csv (包含9个指标)")
    print(f"   - ml_cross_model_comparison.csv (包含9个指标)")

    print(f"\n实验配置:")
    print(f"- 随机划分次数: {n_runs}")
    print(f"- 评估指标数量: 9 (新增: Log Loss, MSE, MAE)")
    print(f"- 模态组合数量: {len(modality_combinations)}")
    print(f"- 机器学习模型: {', '.join(model_names)}")
    print(f"- 总样本数: {len(y_data)}")
    print(f"- 类别数量: {n_classes}")
    print(f"- 总耗时: {elapsed_time:.2f} 秒")

    print(f"\n评估的模态组合:")
    for name, combo in modality_combinations.items():
        print(f"  {name}: {combo}")

    print(f"\n评估的机器学习模型:")
    for model in model_names:
        print(f"  - {model}")

    print("\n新增评估指标说明:")
    print("- Log Loss: 对数损失，衡量概率预测的准确性，越小越好")
    print("- MSE: 均方误差，衡量概率预测与真实标签的误差，越小越好")
    print("- MAE: 平均绝对误差，衡量概率预测与真实标签的绝对误差，越小越好")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 全局变量，用于模型创建
    n_classes = None
    main()