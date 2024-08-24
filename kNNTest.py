import kNN
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