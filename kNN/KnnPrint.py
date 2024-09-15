import kNN
import matplotlib.pyplot as plt
import numpy as np
# 獲取數據
group, labels = kNN.createDataSet()

# 分類數據
x_A = group[np.array(labels) == 'A'][:, 0]
y_A = group[np.array(labels) == 'A'][:, 1]
x_B = group[np.array(labels) == 'B'][:, 0]
y_B = group[np.array(labels) == 'B'][:, 1]

# 繪製點圖
plt.scatter(x_A, y_A, color='blue', label='A')
plt.scatter(x_B, y_B, color='red', label='B')

# 添加標題和標籤
plt.title('Scatter Plot of Data Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 顯示圖形
plt.show()