from numpy import  *
import operator

def createDataSet() :
    #define data set
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0,0],
                   [0,0.1]])
    labels = ['A','A','B','B']
    return group , labels

'''
    kNN 算法的實施（實現方法）-歐式距離
    對未知類別屬性的數據集中的每個點依次執行以下操作：
(1)計算已知類別數據集中的點與當前點的距離；
(2)按照距離遞增次序排序；
(3)選取當前點距離最小的k個點；
(4)確定前k個點所在類別的出現頻率；
(5)返回前k個點出現頻率最高的類別作爲當前點的預測分類。
        Implementation of kNN algorithm (implementation method) - Euclidean distance
    Perform the following operations in sequence for each point in the dataset with unknown categorical attributes:
(1) Calculate the distance between the point in the known category data set and the current point;
(2) Sort in ascending order of distance;
(3) Select k points with the smallest distance from the current point;
(4) Determine the frequency of occurrence of the category where the first k points belong;
(5) Return the category with the highest frequency of occurrence among the first k points as the predicted classification of the current point.
'''
