import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(cm, 
                          classes,
                          title,
                          save_path,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    cm:混淆矩阵值
    classes:分类标签
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
                fontproperties='Times New Roman',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实值',fontsize=14)
    plt.xlabel('预测值',fontsize=14)
    plt.savefig(os.path.join(save_path,"cm.jpg"))


classes=["背景", "建筑","农用地","森林","水体"]
path = r"log\Ablation_Study\DensePPM\files\confusion_matrix-1677789916.1747317.npy"
save_path = os.path.abspath(os.path.dirname(path) + os.path.sep + ".")

cm = np.load(path)
# 删除无用数据
cm = np.delete(cm, 4, axis=0)
cm = np.delete(cm, 4, axis=1)
CMSum = np.sum(cm, axis=1)
cm=cm/CMSum[:,None]
plot_confusion_matrix(cm, classes, '', save_path)

