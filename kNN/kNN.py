from numpy import  *
import numpy as np
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

def classify0 (inX , dataSet ,labels, k):
    '''

    :param inX:Input vector to be classified
    :param dataSet:Input training sample set (vector set)
    :param labels:Label set of sample set
    :param k:Number of nearest neighbors selected
    :return:The label obtained by Knn classification of the input vector
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances =sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortDisIndicies = distances.argsort()
    classCount = {}
    for i in range (k):
        voteIlabel = labels[sortDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount [0][0]

def file2matrix(filename) :
    '''

    :param filename:Data file path
    :return:data vector matrix , label matrix
    '''
    # Open the file
    with open(filename, 'r') as fr:
        # Read all lines from the file
        lines = fr.readlines()

    # Number of lines in the file
    numberOfLines = len(lines)

    # Initialize matrix and class label vector
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []

    # Process each line
    for index, line in enumerate(lines):
        # Strip whitespace and split the line
        line = line.strip()
        listFromLine = line.split('\t')

        # Convert the first three elements to floats and store them in the matrix
        returnMat[index, :] = list(map(float, listFromLine[0:3]))

        # Append the last element as an integer to the class label vector
        classLabelVector.append(int(listFromLine[-1]))

    return returnMat, classLabelVector

#归一化处理
def autoNorm(dataSet):
    '''

    :param dataSet: data set needed to be normalized
    :return:
        normDataSet: normalized data set ,
        ranges: parameter value range
        minVal: minium value of parameter
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet =dataSet-tile(minVals,(m,1))
    normDataSet =normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
