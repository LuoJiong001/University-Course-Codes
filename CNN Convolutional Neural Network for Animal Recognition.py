import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import pickle
# ====================== 1. 加载本地CIFAR-10数据集 ======================
def load_local_cifar10(data_dir):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            train_images.append(data_dict[b'data'])
            train_labels.append(data_dict[b'labels'])

    train_images = np.concatenate(train_images, axis=0)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.concatenate(train_labels, axis=0).reshape(-1, 1)

    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')
        test_images = test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test_dict[b'labels']).reshape(-1, 1)

    return (train_images, train_labels), (test_images, test_labels)


# ====================== 2. 构建模型 ======================
def build_model():
    model = models.Sequential([
        layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                      activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# ====================== 3. 数据加载 ======================
data_dir = "./cifar-10-batches-py"  # 替换为你的数据集路径
(train_images, train_labels), (test_images, test_labels) = load_local_cifar10(data_dir)
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
animal_classes = ['鸟', '猫', '鹿', '狗', '青蛙']

# ====================== 4. 模型权重加载/训练 ======================
model = build_model()
weights_path = "animal_cnn_weights.weights.h5"  # 权重文件路径

if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("✅ 已加载训练好的权重，无需重新训练！")
else:
    print("🚀 未找到权重文件，开始训练模型...")
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        validation_data=(test_images, test_labels)
    )
    model.save_weights(weights_path)
    print(f"✅ 训练完成，权重已保存至：{weights_path}")


# ====================== 5. 图片预处理与预测 ======================
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_single_image(image_path):
    original_img, processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img, verbose=0)
    predicted_prob = tf.nn.softmax(predictions[0])
    predicted_idx = np.argmax(predicted_prob)
    predicted_name = class_names[predicted_idx]
    confidence = predicted_prob[predicted_idx] * 100

    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis('off')
    title = f"识别结果：{'动物-' if predicted_name in animal_classes else '非动物-'}{predicted_name}\n置信度：{confidence:.2f}%"
    plt.title(title, fontsize=12)
    plt.show()
    print(f"图片：{image_path}\n识别结果：{predicted_name}（置信度：{confidence:.2f}%）")


# ====================== 6. GUI选择图片识别 ======================
def select_image_via_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        predict_single_image(file_path)
    else:
        print("未选择图片！")


if __name__ == "__main__":
    select_image_via_gui()