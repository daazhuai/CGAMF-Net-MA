import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from itertools import combinations
import time
import warnings
import os
import json
from tqdm import tqdm

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


# ======================== 数据加载和预处理 ========================
def read_csv_file(path):
    return pd.read_csv(path)

# tcga
# def normalize_sample_id(x):
#     return str(x).split("-01")[0].split("-02")[0]

# metabric
def normalize_sample_id(x):
    return str(x)  # 或者根据需要处理

# 文件路径
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

    def save_model(self, model, modalities, run_id, split_id=0):
        """保存模型"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")
        torch.save(model.state_dict(), cache_path)
        return cache_path

    def load_model(self, model_class, dims, hidden_dim, n_classes, modalities, run_id, split_id=0, device=None):
        """加载模型"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")

        if os.path.exists(cache_path):
            model = model_class(dims, hidden_dim, n_classes)
            model.load_state_dict(torch.load(cache_path))
            if device:
                model.to(device)
            model.eval()
            return model
        return None

    def model_exists(self, modalities, run_id, split_id=0):
        """检查模型是否存在"""
        cache_key = self.get_cache_key(modalities, run_id, split_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pth")
        return os.path.exists(cache_path)


# ======================== 训练函数 ========================
def train_model(modalities_list, train_X_dict, train_y, n_class, hidden_dim=128, epochs=50, lr=0.001):
    """训练单个模型"""
    # 创建模型维度字典
    dims = {}
    for mod in modalities_list:
        if mod in train_X_dict:
            dims[mod] = train_X_dict[mod].shape[1]
        else:
            print(f"错误: 模态 {mod} 在 train_X_dict 中不存在")
            return None

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

    return model


# ======================== 简化版：只训练并保存所有子模型 ========================
def train_and_save_all_models(all_modalities, X_data, y_data, n_classes, n_runs=100, cache_dir="model_cache_train"):
    """
    简化版：只训练并保存所有子模型
    返回: 训练统计信息
    """
    print(f"开始训练 {n_runs} 次划分的所有子模型...")
    print(f"每次划分需要训练的子模型数量: {2 ** len(all_modalities) - 1} = 15")
    print(f"总训练次数: {n_runs} × 15 = {n_runs * 15}")

    # 初始化缓存
    model_cache = ModelCache(cache_dir=cache_dir)

    # 生成所有子模型组合（所有非空子集）
    all_sub_models = []
    for r in range(1, len(all_modalities) + 1):
        for combo in combinations(all_modalities, r):
            all_sub_models.append(list(combo))

    print(f"总共有 {len(all_sub_models)} 种子模型组合")

    # 训练统计信息
    training_stats = {
        'total_runs': n_runs,
        'total_models': n_runs * len(all_sub_models),
        'models_trained': 0,
        'models_loaded': 0,
        'models_failed': 0,
        'run_times': []
    }

    # 进度条
    pbar_runs = tqdm(range(n_runs), desc="训练进度")

    for run_id in pbar_runs:
        random_seed = 42 + run_id
        run_start_time = time.time()

        # 1. 划分训练集和验证集
        total_indices = np.arange(len(y_data))
        train_idx, val_idx = train_test_split(
            total_indices,
            test_size=0.2,
            random_state=random_seed,
            stratify=y_data
        )

        # 2. 标准化数据
        train_X_dict = {}

        for mod in all_modalities:
            X_mod = X_data[mod]

            scaler = StandardScaler()
            X_train = X_mod[train_idx]
            scaler.fit(X_train)

            X_train_std = scaler.transform(X_train)

            train_X_dict[mod] = torch.tensor(X_train_std, dtype=torch.float32)

        train_y = torch.tensor(y_data[train_idx], dtype=torch.long)

        # 3. 训练或加载所有子模型
        run_models_trained = 0
        run_models_loaded = 0

        for sub_model_idx, sub_modalities in enumerate(all_sub_models):
            model = None

            # 检查是否已缓存
            if model_cache.model_exists(sub_modalities, run_id):
                # 尝试加载缓存的模型
                dims = {mod: X_data[mod].shape[1] for mod in sub_modalities}
                try:
                    model = model_cache.load_model(
                        MultiOmicNet, dims, 128, n_classes,
                        sub_modalities, run_id
                    )
                    if model is not None:
                        run_models_loaded += 1
                except:
                    model = None

            # 如果没有缓存或加载失败，训练新模型
            if model is None:
                model = train_model(
                    sub_modalities,
                    train_X_dict,
                    train_y,
                    n_class=n_classes,
                    hidden_dim=128,
                    epochs=50,
                    lr=0.001
                )

                # 保存模型
                if model is not None:
                    model_cache.save_model(model, sub_modalities, run_id)
                    run_models_trained += 1
                else:
                    training_stats['models_failed'] += 1

        # 更新统计信息
        training_stats['models_trained'] += run_models_trained
        training_stats['models_loaded'] += run_models_loaded

        run_time = time.time() - run_start_time
        training_stats['run_times'].append(run_time)

        # 更新进度条
        pbar_runs.set_postfix({
            '当前运行': run_id + 1,
            '本次训练': run_models_trained,
            '本次加载': run_models_loaded,
            '用时': f"{run_time:.1f}s"
        })

    # 计算平均时间
    if training_stats['run_times']:
        training_stats['avg_run_time'] = np.mean(training_stats['run_times'])
        training_stats['total_time'] = np.sum(training_stats['run_times'])
    else:
        training_stats['avg_run_time'] = 0
        training_stats['total_time'] = 0

    return training_stats


# ======================== 主程序 ========================
if __name__ == "__main__":
    print("=" * 80)
    print("简化版：只训练并保存所有子模型")
    print("=" * 80)

    # 设置总随机种子
    set_seed(42)

    # 1. 数据预处理
    print("\n1. 数据预处理...")

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

    # 2. 定义所有模态
    print("\n2. 定义所有模态...")

    # 所有可用模态
    all_modalities = ['clin', 'cnv', 'snv', 'mrna']

    print(f"所有可用模态: {all_modalities}")
    print(f"将生成 {2 ** len(all_modalities) - 1} = 15 种子模型组合")

    # 3. 训练并保存所有模型
    print("\n3. 开始训练并保存所有子模型...")

    start_time = time.time()

    training_stats = train_and_save_all_models(
        all_modalities=all_modalities,
        X_data=X_data,
        y_data=y_data,
        n_classes=n_classes,
        n_runs=100,
        cache_dir="model_cache"
    )

    total_time = time.time() - start_time

    # 4. 保存训练统计信息
    print("\n4. 保存训练统计信息...")

    output_dir = "training_summary"
    os.makedirs(output_dir, exist_ok=True)

    # 保存统计信息为JSON
    stats_json_path = os.path.join(output_dir, "training_statistics.json")
    with open(stats_json_path, 'w') as f:
        json.dump(training_stats, f, indent=2)

    # 保存统计信息为文本
    txt_path = os.path.join(output_dir, "training_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("子模型训练统计摘要\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"实验配置:\n")
        f.write(f"- 随机划分次数: {training_stats['total_runs']}\n")
        f.write(f"- 每次划分的子模型数: {2 ** len(all_modalities) - 1}\n")
        f.write(f"- 总样本数: {len(y_data)}\n")
        f.write(f"- 类别数量: {n_classes}\n")
        f.write(f"- 类别标签: {list(le_y.classes_)}\n")
        f.write(f"- 所有模态: {all_modalities}\n")
        f.write(f"- 缓存目录: model_cache/\n\n")

        f.write(f"训练统计:\n")
        f.write(f"- 总训练次数: {training_stats['total_models']}\n")
        f.write(f"- 实际训练模型数: {training_stats['models_trained']}\n")
        f.write(f"- 加载已有模型数: {training_stats['models_loaded']}\n")
        f.write(f"- 训练失败模型数: {training_stats['models_failed']}\n")
        f.write(f"- 平均每次划分用时: {training_stats.get('avg_run_time', 0):.2f} 秒\n")
        f.write(f"- 总训练用时: {training_stats.get('total_time', 0):.2f} 秒\n")
        f.write(f"- 程序总用时: {total_time:.2f} 秒\n\n")

        if training_stats.get('run_times'):
            f.write(f"每次划分用时详情:\n")
            for i, run_time in enumerate(training_stats['run_times']):
                f.write(f"  划分 {i + 1:3d}: {run_time:.2f} 秒\n")

    print(f"训练统计信息已保存到: {output_dir}")

    # 5. 打印统计摘要
    print("\n" + "=" * 80)
    print("训练统计摘要")
    print("=" * 80)

    print(f"\n实验配置:")
    print(f"- 随机划分次数: {training_stats['total_runs']}")
    print(f"- 每次划分的子模型数: {2 ** len(all_modalities) - 1}")
    print(f"- 总样本数: {len(y_data)}")
    print(f"- 类别数量: {n_classes}")
    print(f"- 类别标签: {list(le_y.classes_)}")
    print(f"- 所有模态: {all_modalities}")
    print(f"- 缓存目录: model_cache/")

    print(f"\n训练统计:")
    print(f"- 总训练次数: {training_stats['total_models']}")
    print(f"- 实际训练模型数: {training_stats['models_trained']}")
    print(f"- 加载已有模型数: {training_stats['models_loaded']}")
    print(f"- 训练失败模型数: {training_stats['models_failed']}")
    print(f"- 平均每次划分用时: {training_stats.get('avg_run_time', 0):.2f} 秒")
    print(f"- 总训练用时: {training_stats.get('total_time', 0):.2f} 秒")
    print(f"- 程序总用时: {total_time:.2f} 秒")

    # 6. 列出生成的模型文件
    print("\n" + "=" * 80)
    print("生成的模型文件示例")
    print("=" * 80)

    cache_dir = "model_cache"
    if os.path.exists(cache_dir):
        model_files = [f for f in os.listdir(cache_dir) if f.endswith('.pth')]
        print(f"\n缓存目录 '{cache_dir}' 中共有 {len(model_files)} 个模型文件")

        if model_files:
            print("\n前10个模型文件示例:")
            for i, file_name in enumerate(model_files[:10]):
                print(f"  {file_name}")

            if len(model_files) > 10:
                print(f"  ... 还有 {len(model_files) - 10} 个文件")

            # 分析文件名模式
            print(f"\n文件名模式分析:")
            sample_file = model_files[0]
            print(f"  示例文件名: {sample_file}")

            # 提取信息
            parts = sample_file.replace('.pth', '').split('_')
            if len(parts) >= 4:
                run_id = parts[1]
                modalities = "_".join(parts[3:])
                print(f"  运行ID: {run_id}")
                print(f"  模态组合: {modalities}")

        # 检查文件数量是否符合预期
        expected_files = training_stats['total_runs'] * (2 ** len(all_modalities) - 1)
        print(f"\n文件数量检查:")
        print(f"  预期文件数: {expected_files}")
        print(f"  实际文件数: {len(model_files)}")

        if len(model_files) == expected_files:
            print("  ✓ 文件数量符合预期")
        else:
            print(f"  ⚠ 文件数量不符，可能有些模型训练失败")
    else:
        print(f"缓存目录 '{cache_dir}' 不存在")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print("\n后续步骤建议:")
    print("1. 使用训练好的子模型进行模型平均评估")
    print("2. 模型保存在: model_cache/ 目录中")
    print("3. 可以使用以下代码加载模型:")
    print("""
    model_cache = ModelCache(cache_dir="model_cache")
    dims = {mod: X_data[mod].shape[1] for mod in modalities}
    model = model_cache.load_model(MultiOmicNet, dims, 128, n_classes, modalities, run_id)
    """)