import numpy as np
import os
import kNN


# 将图像文件转换为向量
def img2Vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 数字识别器的运行函数
def digitRecognizerRunner():
    hwLabels = []
    # 获取训练数据集
    trainingFileList = os.listdir('/digitRecognizer/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))

    # 处理每一个训练文件
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉文件扩展名
        classNumStr = int(fileStr.split('_')[0])  # 获取分类号
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector(f'/digitRecognizer/trainingDigits/{fileNameStr}')  # 读取图像文件

    # 获取测试数据集
    testFileList = os.listdir('/digitRecognizer/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    # 测试每一个测试文件
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉文件扩展名
        classNumStr = int(fileStr.split('_')[0])  # 获取真实分类号
        vectorUnderTest = img2Vector(f'/digitRecognizer/testDigits/{fileNameStr}')  # 读取测试图像文件

        # 通过 kNN 进行分类
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f"the classifier came back with: {classifierResult}, the real answer is: {classNumStr}")

        # 统计错误数量
        if classifierResult != classNumStr:
            errorCount += 1.0

    # 输出错误数和错误率
    print(f"\nthe total number of errors is: {errorCount}")
    print(f"the error rate is: {errorCount / float(mTest)}")
