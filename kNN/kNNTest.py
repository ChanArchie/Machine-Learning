import kNN
from numpy import array
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels =file2matrix('datingTestSet.txt')
    normMat,ranges,minVals =autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount =0.0
    for i in range (numTestVecs):
        classifierResult = kNN.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print(f"the classifier came back with: {classifierResult} ,the real answer is: {datingLabels[i]}")
        if(classifierResult != datingLabels[i]):
            errorCount +=1.0
    print(f"the total error rate is: {errorCount/float(numTestVecs)}")

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats= float(input("percentage of time spent playing into video game?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels =kNN.file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals =kNN.autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult =kNN.classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:"+resultList[classifierResult-1])

