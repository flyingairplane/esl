###esl第四章fig4.13
###l1-logistic regression
###Date:2020/08/24
###令tol=1e-3时，曲线不光滑，在lambda位于区间[0.05,0.07]之间系数出现跳跃，分析原因，lambda在区间外迭代了4次，在区间内迭代了5次，
###在区间内迭代第四次后tol接近tol，因此需要多迭代一次，降低tol后曲线变得平滑。
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

##IRLS每次迭代后使用LARS
#x,输入
#y,响应
#lam,惩罚因子
def countbeta(x,y,lam):
    tol=1e-6
    maxiter=1000
    residual=np.inf
    iter=0
    lasterror=np.inf
    beta=np.zeros((x.shape[1],))
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
    return beta.reshape((x.shape[1],1))

##IRLS
#x,y为训练集的输入输出
#lambda为惩罚项系数
def deallam(x,y,lam):
    featurelen=x.shape[1]-1
    Nlen=x.shape[0]
    beta_old=np.ones((featurelen+1,1))
    beta_new = np.zeros((featurelen + 1, 1))
    tol=1e-6
    iternum=1000#最大迭代次数
    iter=0
    W=np.zeros((featurelen+1,featurelen+1))
    while np.mean(np.square(beta_new-beta_old))>tol and iter<iternum:
        iter=iter+1
        print(iter)
        beta_old=beta_new
        p=np.exp(x.dot(beta_old))/(1+np.exp(x.dot(beta_old)))
        W=np.diag((p*(1-p))[:,0])
        z=x.dot(beta_old)+np.diag(1/((p*(1-p))[:,0])).dot(np.array((y-p[:,0])).reshape((Nlen,1)))

        tx = np.diag(np.sqrt(np.diag(W))).dot(x)
        ty = np.diag(np.sqrt(np.diag(W))).dot(z)
        beta_new = countbeta(tx, ty[:,0], lam)
    if iter==iternum:
        print('not convergence.')
    return beta_new


currentfolder='D:\\Program\\python\\esl\\chapter4\\'
totaldata=pd.read_csv(currentfolder+'SAheart.data.txt',delimiter=',',header=0,index_col=0)
xdata=totaldata.loc[:,['sbp','tobacco','ldl','famhist','obesity','alcohol','age']]
Nlen=totaldata.shape[0]
featurelen = xdata.shape[1]
ydata=totaldata.iloc[:,totaldata.shape[1]-1]
#均值为0，方差为1
ss = preprocessing.StandardScaler()
xdata_ss = ss.fit_transform(xdata)
xdata_ss=np.hstack((np.ones(shape=(len(xdata_ss),1)),xdata_ss))#加一列全1

#惩罚项
lams=np.linspace(1e-3,1,500)#lambda
betas=np.zeros(shape=(featurelen+1,len(lams)))
for i in range(0,len(lams)):
    print("deal lambda:"+str(lams[i]))
    beta0=deallam(xdata_ss, ydata, lams[i])
    betas[:,i]=beta0[:,0]

betas=betas[1:,:]#排除截距
#x轴
betasum=np.sum(np.abs(betas),axis=0)
#label
labels=xdata.columns.values

#绘图
plt.style.use('ggplot')
fig1, ax1 = plt.subplots()
lw=2
ax1.plot(betasum,betas.T,'-x')
for i in range(0,featurelen):
    ax1.text(np.max(betasum),betas[i,0], labels[i])
ax1.set_xlabel(r'$||\beta(\lambda)||$')
ax1.set_ylabel('coefs')
#ax1.set_xlim((0, 2.5))
ax1.set_title('L1 regularized logistic regression')
plt.show()