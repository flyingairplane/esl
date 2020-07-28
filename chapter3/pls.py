###esl第三章fig3.7
###partial least squares
###Date:2020/07/27
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def cvsplit(n, k):  # n为训练集样本数,k为k-fold
    testX = []
    trainX = []
    eachnum = round(n / k)
    foldnum = eachnum * np.ones((k,))
    if eachnum * k > n:
        index = np.sort(np.random.choice(np.arange(0, k), replace=False, size=eachnum * k - n))
        foldnum[index] = foldnum[index] - 1
    elif eachnum * k < n:
        index = np.sort(np.random.choice(np.arange(0, k), replace=False, size=n - eachnum * k))
        foldnum[index] = foldnum[index] + 1
    for i in range(0, k):
        index = list(np.arange(np.sum(foldnum[0:i]), np.sum(foldnum[0:i]) + foldnum[i]))
        index = [int(i) for i in index]
        testX.append(index)
        trainX.append(list(set(np.arange(0, n)) - set(index)))
    return trainX, testX
#fold为交叉验证折数
#x,y为训练集的输入输出
#traininindex,testindex分别为训练集(training set)和验证集(validation set)的索引
#df为主成分数量
def countMSE(fold,x,y,trainindex,testindex,df):
    mses=np.zeros((fold,))
    for i in range(0,fold):
        txtrain = x[trainindex[i], :]
        txtrain = txtrain - np.mean(txtrain, axis=0)  # 10折之后均值不为0，再次零均值化
        tx=txtrain.copy()
        txtest = x[testindex[i], :]
        tytrain = y.iloc[trainindex[i]]
        ty = tytrain - np.mean(tytrain)
        if df==0:
            typredict =np.mean(tytrain)
        else:
            ym=np.zeros(shape=(tx.shape[0],))
            for m in range(0,df):
                zm=0
                for j in range(0,x.shape[1]):
                    faimj=tx[:,j].T.dot(ty)
                    zm=zm+faimj*tx[:,j]
                thetam=zm.T.dot(ty)/(zm.T.dot(zm))
                ym=ym+thetam*zm
                for j in range(0,x.shape[1]):
                    tx[:,j]=tx[:,j]-zm.T.dot(tx[:,j])/(zm.T.dot(zm))*zm
            betahat=np.linalg.inv(txtrain.T.dot(txtrain)).dot(txtrain.T).dot(ym)
            typredict = txtest.dot(betahat) + np.mean(tytrain)-np.mean(x[trainindex[i], :], axis=0).dot(betahat)
        mses[i] = np.mean(np.square(y.iloc[testindex[i]] - typredict))

    return np.mean(mses), np.std(mses)/np.sqrt(fold)

currentfolder='D:\\Program\\python\\esl\\Prostate\\'
xlsx = pd.ExcelFile(currentfolder+'ProstateData.xlsx')
totaldata=pd.read_excel(xlsx, sheet_name='Sheet1',header=0,index_col=0)

#train data
traindata=totaldata[totaldata.train=='T']
traindata=traindata.iloc[:,0:9]
txdata=traindata.iloc[:,0:8]
tydata=traindata.iloc[:,8]

Nlen=txdata.shape[0]
featurelen=txdata.shape[1]
fold=10#tenfold cv

#均值为0，方差为1
ss = preprocessing.StandardScaler()
txdata_ss = ss.fit_transform(txdata)

#pcr
dfs=[0,1,2,3,4,5,6,7,8]#主成分数量
trainind,testind=cvsplit(Nlen,fold)#10-fold
ridgecverror=np.zeros(shape=(len(dfs),))
ridgese=np.zeros(shape=(len(dfs),))
for i in range(0,len(dfs)):
    print("deal df:"+str(dfs[i]))
    ridgecverror[i],ridgese[i]=countMSE(fold, txdata_ss, tydata, trainind, testind, dfs[i])

plt.style.use('ggplot')
fig, ax = plt.subplots()
lw=2
plt.errorbar(np.arange(0,len(ridgecverror)), ridgecverror, yerr = ridgese, fmt='o',ls='-', capsize=4, capthick=2)
plt.plot(np.linspace(0,len(ridgecverror),100),np.ones(shape=(100,1))*np.min(ridgecverror+ridgese),'-.')
plt.xlabel('Number of Directions')
plt.ylabel('CV Error')
plt.title('PLS')
plt.show()