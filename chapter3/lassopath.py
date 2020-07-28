###esl第三章fig3.10
###lasso path
###使用lar算法求lasso path
###Date:2020/07/25
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

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
x = ss.fit_transform(txdata)

#lasso path
y=tydata-np.mean(tydata)
#x为预测因子
#y为响应
activeset=[]
unactiveset=np.arange(0,featurelen)#和active set互补
cor=x.T.dot(y)
activeset.append(np.argmax(np.abs(cor)))
unactiveset=np.delete(unactiveset,np.array(activeset))
insetnum=1
betapath=np.zeros(shape=(featurelen,featurelen+1))#存放每步的系数
deltak=0
rk=y#残差
for k in range(1,featurelen+1):
    beta=betapath[:,k-1].copy()#不用copy会导致betapath被修改
    XAk=x[:,activeset]
    deltak = np.linalg.inv(XAk.T.dot(XAk)).dot(XAk.T).dot(rk)
    #存放每个候选因子的相关值
    cors=np.zeros(shape=(len(unactiveset),))
    # 存放每个候选因子对应的alpha
    alphas=np.zeros(shape=(len(unactiveset),))
    for j in range(0,len(unactiveset)):
        bi=unactiveset[j]
        ai=activeset[0]
        alpha=(x[:,bi].T.dot(rk)-x[:,ai].T.dot(rk))/(x[:,bi].T.dot(XAk).dot(deltak)-x[:,ai].T.dot(XAk).dot(deltak))
        if alpha<=0 or alpha>1:#alpha值只能在(0,1]之间
            alpha=(x[:,bi].T.dot(rk)+x[:,ai].T.dot(rk))/(x[:,bi].T.dot(XAk).dot(deltak)+x[:,ai].T.dot(XAk).dot(deltak))
        alphas[j]=alpha
        rkp1 =y-XAk.dot(beta[activeset])-np.asarray(alpha*XAk.dot(deltak).reshape((Nlen))) # r(k+1)
        cors[j]=np.abs(x[:,bi].T.dot(rkp1))
    if np.size(cors)!=0:
        addj=np.argmax(cors)
        rk=y-XAk.dot(beta[activeset])-np.asarray(alphas[addj]*XAk.dot(deltak).reshape((Nlen)))
        beta[activeset]=beta[activeset]+alphas[addj]*deltak
        betapath[:,k]=beta.copy()
        activeset.append(unactiveset[addj])
        unactiveset = np.delete(unactiveset, addj)
    else:#最后一个因子加入，alpha=1
        beta[activeset] = beta[activeset] + 1.0 * deltak
        betapath[:, k] = beta.copy()

#x轴
betasum=np.sum(np.abs(betapath),axis=0)
betasum=betasum/np.max(betasum)
#label
labels=txdata.columns.values

#绘图
plt.style.use('ggplot')
fig1, ax1 = plt.subplots()
lw=2
ax1.plot(betasum,betapath.T,'-x')
for i in range(0,featurelen):
    ax1.text(1,betapath[i,featurelen-1], labels[i])
ax1.set_xlabel('shrinkage factor s')
ax1.set_ylabel('coefs')
ax1.set_xlim((0, 1.2))
ax1.set_title('lasso path')
plt.show()

