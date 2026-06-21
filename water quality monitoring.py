# -*- coding: utf-8 -*-
# 二值化处理 + 颜色矩特征提取 + SVM水质分类
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # 新增：特征标准化（提升SVM效果）

# ========== 配置参数（跨平台兼容） ==========
path = r'./图片'  # macOS/Linux路径
# path = r'F:\新教材资料\水色图像水质评价\图片'  # Windows路径（备用）
target_crop_size = 100  # 中心裁剪100x100区域
valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')  # 支持的图片格式

# ========== 第一步：安全加载图片并提取颜色矩特征 ==========
# 过滤有效图片文件（避免非图片文件干扰）
file_list = []
for fname in os.listdir(path):
    if fname.lower().endswith(valid_ext):
        file_list.append(fname)

if not file_list:
    raise ValueError(f"错误：在 {path} 文件夹中未找到有效图片！")

# 预定义特征矩阵和标签（动态适配有效文件数）
n_samples = len(file_list)
X = np.zeros((n_samples, 9), dtype=np.float32)  # 9个颜色矩特征
Y = np.zeros(n_samples, dtype=np.int32)  # 水质类别标签

# 遍历处理每张图片
valid_indices = []  # 记录有效样本索引（避免错误样本污染数据）
for i, img_name in enumerate(file_list):
    try:
        # 1. 跨平台路径拼接（核心修复：替代\\）
        img_path = os.path.join(path, img_name)
        # 2. 读取图片并确保RGB模式（避免灰度/透明通道问题）
        img = Image.open(img_path).convert('RGB')
        im = img.split()  # 分离RGB通道

        # 3. 处理R通道并计算裁剪范围（防越界）
        R = np.array(im[0], dtype=np.float32) / 255.0
        h, w = R.shape
        half_size = target_crop_size // 2
        # 边界保护：如果图片尺寸不足100x100，直接缩放
        if h < target_crop_size or w < target_crop_size:
            img_resized = img.resize((target_crop_size, target_crop_size), Image.Resampling.LANCZOS)
            im_resized = img_resized.split()
            R = np.array(im_resized[0], dtype=np.float32) / 255.0
            G = np.array(im_resized[1], dtype=np.float32) / 255.0
            B = np.array(im_resized[2], dtype=np.float32) / 255.0
        else:
            # 计算中心裁剪坐标（避免负索引）
            row_1 = max(0, int(h / 2) - half_size)
            row_2 = min(h, int(h / 2) + half_size)
            con_1 = max(0, int(w / 2) - half_size)
            con_2 = min(w, int(w / 2) + half_size)
            # 裁剪中心区域
            R = R[row_1:row_2, con_1:con_2]
            G = np.array(im[1], dtype=np.float32) / 255.0
            G = G[row_1:row_2, con_1:con_2]
            B = np.array(im[2], dtype=np.float32) / 255.0
            B = B[row_1:row_2, con_1:con_2]

        # 4. 计算颜色矩特征（优化数值稳定性）
        # 一阶矩（均值）
        r1 = np.mean(R)
        g1 = np.mean(G)
        b1 = np.mean(B)

        # 二阶矩（标准差）
        r2 = np.std(R, ddof=1)  # ddof=1：样本标准差（更贴合实际）
        g2 = np.std(G, ddof=1)
        b2 = np.std(B, ddof=1)

        # 三阶矩（偏度的立方根）
        r_mean = R.mean()
        g_mean = G.mean()
        b_mean = B.mean()
        r3 = np.cbrt(np.mean(np.abs(R - r_mean) ** 3))  # 用np.cbrt替代**(1/3)，避免负数问题
        g3 = np.cbrt(np.mean(np.abs(G - g_mean) ** 3))
        b3 = np.cbrt(np.mean(np.abs(B - b_mean) ** 3))

        # 5. 赋值到特征矩阵
        X[i] = [r1, g1, b1, r2, g2, b2, r3, g3, b3]

        # 6. 提取标签（鲁棒处理）
        I = img_name.find('_')
        if I == -1 or not img_name[:I].isdigit():
            raise ValueError(f"文件名格式错误：{img_name}（缺少_或前缀非数字）")
        label = int(img_name[:I])
        Y[i] = label
        valid_indices.append(i)

        print(f"成功处理：{img_name} → 标签{label}，R一阶矩：{r1:.4f}")

    except Exception as e:
        print(f"警告：跳过 {img_name} → {e}")

# 过滤仅保留有效样本（避免错误样本影响训练）
X = X[valid_indices]
Y = Y[valid_indices]
if len(X) < 2:
    raise ValueError("有效样本数不足2个，无法进行训练/测试拆分！")

# ========== 第二步：数据预处理（提升SVM性能） ==========
# 特征标准化（SVM对特征尺度敏感，必须做！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练集/测试集
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=4, stratify=Y  # stratify：保持类别分布
)

# ========== 第三步：SVM模型训练与评估 ==========
# 优化SVM参数（提升分类效果）
clf = SVC(
    kernel='rbf',  # 径向基核函数（比默认更适合非线性特征）
    class_weight='balanced',
    C=1.0,  # 正则化参数（可调整：0.1/1/10）
    gamma='scale',  # 核函数系数（自动适配特征尺度）
    random_state=4
)

# 训练模型（原代码中X*40无依据，移除并使用标准化特征）
clf.fit(x_train, y_train)

# 预测与评估
y1 = clf.predict(x_test)
correct = np.sum(y1 == y_test)
accuracy = correct / len(y_test)

print("\n========== 模型评估结果 ==========")
print(f"测试集样本数：{len(y_test)}")
print(f"预测正确数：{correct}")
print(f"预测准确率：{accuracy:.4f}")

# 可选：保存模型和标准化器（方便后续预测）
import joblib

joblib.dump(clf, 'water_quality_svm.pkl')
joblib.dump(scaler, 'water_quality_scaler.pkl')
print("\n模型已保存：water_quality_svm.pkl")
print("标准化器已保存：water_quality_scaler.pkl")