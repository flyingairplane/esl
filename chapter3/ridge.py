###esl第三章fig3.7
###ridge
###Date:2020/07/21
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
#df为自由度，根据df可以算出lambda
def countMSE(fold,x,y,trainindex,testindex,df):
    mses=np.zeros((fold,))
    for i in range(0,fold):
        txtrain = x[trainindex[i], :]
        txtrainstd = txtrain - np.mean(txtrain, axis=0)  # 10折之后均值不为0，再次零均值化
        txtest = x[testindex[i], :]
        ty = y.iloc[trainindex[i]]
        tycenter = ty - np.mean(ty)
        lam = countlambda(txtrainstd, df)
        u, s, vh = np.linalg.svd(txtrainstd)
        betahat = vh.T.dot(np.diag(s / (s ** 2 + lam))).dot(u[:, 0:x.shape[1]].T).dot(tycenter)
        #
        # betahat = np.linalg.inv(txtrain.T.dot(txtrain)+lam*np.eye(x.shape[1])).dot(txtrain.T).dot(tycenter)
        # typredict = txtest.dot(betahat)
        # mses[i]=np.mean(np.square(y.iloc[testindex[i]]-np.mean(ty) - typredict))
        typredict = txtest.dot(betahat) + np.mean(ty)-np.mean(txtrain, axis=0).dot(betahat)
        mses[i] = np.mean(np.square(y.iloc[testindex[i]] - typredict))

    return np.mean(mses), np.std(mses)/np.sqrt(fold)

def countlambda(x,df):
    _,sigma,_=np.linalg.svd(x)
    lam=0
    step=5#初始步进
    flag=-1#记录上一次移动的方向，若两次移动方向相同，步进翻倍，若两次移动方向相反，步进减半
    dist=df-countdf(x.shape[1],sigma,lam)
    while np.abs(dist)>0.01:
        if dist<0:
            if flag==-1:
                step=step*2
                lam=lam+step
                dist = df - countdf(x.shape[1], sigma, lam)
            else:
                step=step/2
                lam=lam+step
                dist = df - countdf(x.shape[1], sigma, lam)
                flag=1
        else:
            if flag==1:
                step = step * 2
                lam=lam-step
                dist = df - countdf(x.shape[1], sigma, lam)
            else:
                step = step / 2
                lam = lam - step
                dist = df - countdf(x.shape[1], sigma, lam)
                flag = 1
    return lam

def countdf(p,dj,lam):
    sum=0
    for i in range(0,p):
        sum=sum+dj[i]**2/(dj[i]**2+lam)
    return sum

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

#ridge regression
dfs=[0.01,1,2,3,4,5,6,7,8]#df=0对应着lambda=inf，无法实现
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
plt.xlabel('Degree of Freedom')
plt.ylabel('CV Error')
plt.title('Ridge')
plt.show()

#sklearn
srcverror=np.zeros(shape=(len(dfs),))
srse=np.zeros(shape=(len(dfs),))
cv = KFold(n_splits=fold)
for i in range(0,len(dfs)):
    mses = np.zeros((fold,))
    for j, (train, test) in enumerate(cv.split(txdata_ss, tydata)):
        lam = countlambda(txdata_ss[train], dfs[i])
        rg=Ridge(alpha=lam)
        rg.fit(txdata_ss[train], tydata.iloc[train])
        y_predict=rg.predict(txdata_ss[test])
        mses[j] = np.mean(np.square(y_predict - tydata.iloc[test]))

    srcverror[i]=np.mean(mses)
    srse[i]=np.std(mses) / np.sqrt(fold)

plt.style.use('ggplot')
fig, ax = plt.subplots()
lw=2
plt.errorbar(np.arange(0,len(srcverror)), srcverror, yerr = srse, fmt='o',ls='-', capsize=4, capthick=2)
plt.plot(np.linspace(0,len(srcverror),100),np.ones(shape=(100,1))*np.min(srcverror+srse),'-.')
plt.xlabel('Degree of Freedom')
plt.ylabel('CV Error')
plt.title('Ridge')
plt.show()
