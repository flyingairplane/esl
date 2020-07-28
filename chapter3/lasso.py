###esl第三章fig3.7
###lasso
###Date:2020/07/24
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

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
#x,预测因子
#y,响应
#lam,惩罚因子
#beta,系数的初始值，可以用做warm start
def countbeta(x,y,lam,beta):
    tol=1e-4
    maxiter=1000
    residual=np.inf
    iter=0
    lasterror=np.inf
    while residual>tol:
        for i in range(0,x.shape[1]):
            yresidual=y-x[:,list(set(np.arange(x.shape[1])) - set([i]))].dot(beta[list(set(np.arange(len(beta))) - set([i]))])
            #单变量ls解
            lsi=1/(x[:,i].T.dot(x[:,i]))*x[:,i].T.dot(yresidual)
            betai=np.sign(lsi)*(np.abs(lsi)-lam) if np.abs(lsi)-lam>0 else 0
            beta[i]=betai
        residual=np.abs(np.mean((y-x.dot(beta))**2)-lasterror)
        lasterror=np.mean((y-x.dot(beta))**2)
        iter=iter+1
        if iter>maxiter:
            print("lambda="+str(lam)+": don't convergence.")
            break
    return beta



#fold为交叉验证折数
#x,y为训练集的输入输出
#traininindex,testindex分别为训练集(training set)和验证集(validation set)的索引
#lambda为惩罚项系数
#betahat为warm start
def countMSE(fold,x,y,trainindex,testindex,lam,betahat):
    mses=np.zeros((fold,))
    for i in range(0,fold):
        txtrain = x[trainindex[i], :]
        txtrainstd = txtrain - np.mean(txtrain, axis=0)  # 10折之后均值不为0，再次零均值化
        txtest = x[testindex[i], :]
        ty = y.iloc[trainindex[i]]
        tycenter = ty - np.mean(ty)
        #首次启动的beta初始值为ls解
        if np.size(betahat)==0:
            betahat=np.linalg.inv(txtrainstd.T.dot(txtrainstd)).dot(txtrainstd.T).dot(tycenter)
        betahat = countbeta(txtrainstd, tycenter,lam,betahat)
        typredict = txtest.dot(betahat) + np.mean(ty)-np.mean(txtrain, axis=0).dot(betahat)
        mses[i] = np.mean(np.square(y.iloc[testindex[i]] - typredict))

    return np.mean(mses), np.std(mses)/np.sqrt(fold),betahat


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

#lasso
lams=np.logspace(-2,0,10)#lambda
trainind,testind=cvsplit(Nlen,fold)#10-fold
lassocverror=np.zeros(shape=(len(lams),))
lassose=np.zeros(shape=(len(lams),))
beta0=[]
for i in range(0,len(lams)):
    print("deal lambda:"+str(lams[i]))
    #上一个lambda计算得到的beta作为下一次的warm start
    lassocverror[i],lassose[i],beta0=countMSE(fold, txdata_ss, tydata, trainind, testind, lams[i],beta0)

plt.style.use('ggplot')
fig1, ax1 = plt.subplots()
lw=2
ax1.errorbar(lams, lassocverror, yerr = lassose, fmt='o',ls='-', capsize=4, capthick=2)
ax1.plot(np.linspace(0,np.max(lams),100),np.ones(shape=(100,1))*np.min(lassocverror+lassose),'-.')
ax1.set_xscale("log")#横坐标为对数坐标系
ax1.invert_xaxis()#横坐标水平翻转
ax1.set_xlabel('Lambda')
ax1.set_ylabel('CV Error')
ax1.set_title('Lasso')
#plt.show()

#sklearn
srcverror=np.zeros(shape=(len(lams),))
srse=np.zeros(shape=(len(lams),))
cv = KFold(n_splits=fold)
for i in range(0,len(lams)):
    mses = np.zeros((fold,))
    for j, (train, test) in enumerate(cv.split(txdata_ss, tydata)):
        lasso=Lasso(alpha=lams[i])
        lasso.fit(txdata_ss[train], tydata.iloc[train])
        y_predict=lasso.predict(txdata_ss[test])
        mses[j] = np.mean(np.square(y_predict - tydata.iloc[test]))

    srcverror[i]=np.mean(mses)
    srse[i]=np.std(mses) / np.sqrt(fold)

plt.style.use('ggplot')
fig2, ax2 = plt.subplots()
lw=2
ax2.errorbar(lams, srcverror, yerr = srse, fmt='o',ls='-', capsize=4, capthick=2)
ax2.plot(np.linspace(0,np.max(lams),100),np.ones(shape=(100,1))*np.min(srcverror+srse),'-.')
ax2.set_xscale("log")#横坐标为对数坐标系
ax2.invert_xaxis()#横坐标水平翻转
ax2.set_xlabel('Lambda')
ax2.set_ylabel('CV Error')
ax2.set_title('Lasso')
plt.show()
