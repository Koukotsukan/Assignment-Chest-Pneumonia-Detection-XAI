from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report

from models.model_loader import load_vgg16, load_mobilenet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 定义函数加载数据
def load_data(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = load_img(img_path, target_size=target_size)  # 加载图像
                img_array = img_to_array(img)  # 转为 NumPy 数组
                images.append(img_array)
                labels.append(0 if label.lower() == "normal" else 1)  # 假设分类是 "NORMAL" 和 "PNEUMONIA"
    return np.array(images), np.array(labels)



vgg = load_vgg16()
mobilenet = load_mobilenet()
models = [vgg, mobilenet]

def evaluate_model(model, X_test, y_test):
    # 模型预测
    y_pred = model.predict(X_test)

    # 如果是分类任务
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # 多分类：取概率最大的类别
        y_pred_classes = y_pred.argmax(axis=-1)
    else:
        # 二分类：将概率转为0或1
        y_pred_classes = (y_pred > 0.5).astype(int)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    # roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr') if y_pred.ndim > 1 else roc_auc_score(y_test, y_pred)

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    # print(f"AUC-ROC: {roc_auc:.4f}")

    # 显示分类报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred_classes)

    # 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # 加载训练、验证、测试集
    train_dir = r"C:\Users\ngaus\iCloudDrive\UM M.AI\2024-2025 S1\WQF7009_XAI\Alternative Assessment Pt.1\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\train"
    test_dir = r"C:\Users\ngaus\iCloudDrive\UM M.AI\2024-2025 S1\WQF7009_XAI\Alternative Assessment Pt.1\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\test"
    val_dir = r"C:\Users\ngaus\iCloudDrive\UM M.AI\2024-2025 S1\WQF7009_XAI\Alternative Assessment Pt.1\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\val"

    X_test, y_test = load_data(test_dir, target_size=(150, 150))
    X_test = X_test / 255.0

    print("----------------------------------------------")
    print("Evaluating Model:", vgg.name)
    evaluate_model(vgg, X_test, y_test)
    print("----------------------------------------------")

    X_test, y_test = load_data(test_dir, target_size=(224, 224))
    X_test = X_test / 255.0

    print("----------------------------------------------")
    print("Evaluating Model:", mobilenet.name)
    evaluate_model(mobilenet, X_test, y_test)
    print("----------------------------------------------")