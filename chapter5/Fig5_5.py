##figure 5.5
##2020/09/21
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#for natural cubic spline,delete constant
def dkX(x,k,K):
    return ((x - k > 0) * 1.0 * np.power(x - k, 3)-(x - K > 0) * 1.0 * np.power(x - K, 3))/(K-k)
def truncated_power_basis(x,knots):
    K=len(knots)
    H=np.zeros((len(x),K))
    H[:,0]=1
    H[:,1]=x
    for k in range(1,K-2+1):
        H[:,k+1]=dkX(x,knots[k-1],knots[K-1])-dkX(x,knots[K-2],knots[K-1])
    #return H[:,1:]
    return H
#
#logistic regression
def lr(xtrain,ydata):
    Nlen=xtrain.shape[0]
    beta_old = np.ones((xtrain.shape[1], 1))
    beta_new = np.zeros((xtrain.shape[1], 1))
    tol = 1e-6
    iternum = 1000  # 最大迭代次数
    iter = 0
    W = np.zeros((xtrain.shape[1], xtrain.shape[1]))
    while np.sum(np.square(beta_new - beta_old)) > tol and iter < iternum:
        iter = iter + 1
        #print(iter)
        beta_old = beta_new
        p = np.exp(xtrain.dot(beta_old)) / (1 + np.exp(xtrain.dot(beta_old)))
        W = np.diag((p * (1 - p))[:, 0])
        z = xtrain.dot(beta_old) + np.diag(1 / ((p * (1 - p))[:, 0])).dot(
            np.array((ydata - p[:, 0])).reshape((Nlen, 1)))
        beta_new = np.linalg.inv(xtrain.T.dot(W).dot(xtrain)).dot(xtrain.T).dot(W).dot(z)
    if iter == iternum:
        print('not convergence.')
    return beta_new,W


currentfolder='D:\\Program\\python\\esl\\chapter5\\'
totaldata=pd.read_csv(currentfolder+'phoneme.data.txt',delimiter=',',header=0,index_col=0)
del totaldata['speaker']
dealdata=totaldata[totaldata.g.isin(['aa','ao'])]
dealdata['g']=((dealdata['g']=='aa')*1.0)

trainindex=np.random.choice(np.arange(0,dealdata.shape[0]),1000,replace=False)
trainindex.sort()
testindex=list(set(np.arange(0,dealdata.shape[1]))-set(trainindex))
traindata=dealdata.iloc[trainindex,:]
testdata=dealdata.iloc[testindex,:]


featurelen=dealdata.shape[1]-1
trainx=traindata.iloc[:,0:featurelen]
trainy=traindata.iloc[:,featurelen]
testx=testdata.iloc[:,0:featurelen]
testy=testdata.iloc[:,featurelen]

#
# displaynum=2
# plotaa=trainx[trainy==1]
# plotao=trainx[trainy==0]
# fig, ax = plt.subplots()
# for i in np.arange(0,displaynum):
#     ax.plot(np.arange(1,257),plotaa.iloc[i,:],color='b')
#     ax.plot(np.arange(1,257),plotao.iloc[i,:],color='r')
# ax.set_xlabel('frequency')
# ax.set_ylabel('log-periodogram')
# ax.set_title('phoneme Examples')
# plt.show()
#


theta,_=lr(np.array(trainx),trainy)

classifytrain=np.array(trainx.dot(theta)>0)[:,0]
trainerror=np.count_nonzero(classifytrain!=trainy)/len(trainy)
classifytest=np.array(testx.dot(theta)>0)[:,0]
testerror=np.count_nonzero(classifytest!=testy)/len(testy)

#使用12个knots，使用natural cubic spline
knots=[20,40,60,80,100,120,140,160,180,200,220,240]
H=truncated_power_basis(np.arange(1,257),knots)
thetasmooth,_=lr(np.array(trainx).dot(H),trainy)
betasmooth=H.dot(thetasmooth)

classifytrainsmooth=np.array(trainx.dot(betasmooth)>0)[:,0]
trainerrorsmooth=np.count_nonzero(classifytrainsmooth!=trainy)/len(trainy)
classifytestsmooth=np.array(testx.dot(betasmooth)>0)[:,0]
testerrorsmooth=np.count_nonzero(classifytestsmooth!=testy)/len(testy)
print('train error:'+str(trainerror))
print('test error:'+str(testerror))
print('train smooth error:'+str(trainerrorsmooth))
print('test smooth error:'+str(testerrorsmooth))

fig, ax = plt.subplots()
ax.plot(np.arange(1,257),theta[:,0],color='b')
ax.plot(np.arange(1,257),betasmooth[:,0],color='r')
ax.grid()
ax.set_xlabel('frequency')
ax.set_ylabel('Logistic Regression Coefficients')
ax.set_title('Phoneme Classification Comparison')
plt.show()