##figure 5.11
##2020/09/24
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import seaborn as sns

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

#同时返回每维的结点
#KNs为根据训练集得到的结点
def compute_basis(xtrain,KNs=[]):
    Nlen=xtrain.shape[0]
    xexpand = np.zeros((Nlen, 3 * xtrain.shape[1] + 1))
    xexpand[:, 0] = 1
    if len(KNs)==0:#训练集
        KNs=np.zeros((4,xtrain.shape[1]))
        for i in range(0, xtrain.shape[1]):
            tempx = xtrain[:, i]
            # 因为要返回索引，因此需要稳定排序
            index = np.argsort(tempx, kind='stable')
            tempxsort = tempx[index]
            uniquex= np.unique(tempxsort)
            # knots=np.quantile(uniquex,[0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
            knots = [np.min(uniquex)]
            knots.extend(np.quantile(uniquex, [0.33, 0.67]))
            knots.extend([np.max(uniquex)])
            KNs[:,i]=knots
            xexpand[:, i * 3+1:(i+1) * 3+1] = truncated_power_basis(xtrain[:,i], KNs[:,i])[:,1:]
    else:#测试集
        for i in range(0, xtrain.shape[1]):
            xexpand[:, i * 3+1:(i+1) * 3+1] = truncated_power_basis(xtrain[:,i], KNs[:,i])[:,1:]

    return xexpand,KNs

#tensor product
#仅限二维
def compute_tensor_basis(xtrain,KNs=[]):
    Nlen=xtrain.shape[0]
    xexpand = np.zeros((Nlen, 4 * 4))
    xex=np.zeros((Nlen, 8))
    if len(KNs)==0:
        KNs = np.zeros((4, xtrain.shape[1]))
        for i in range(0, xtrain.shape[1]):
            tempx = xtrain[:, i]
            # 因为要返回索引，因此需要稳定排序
            index = np.argsort(tempx, kind='stable')
            tempxsort = tempx[index]
            uniquex, indices = np.unique(tempxsort, return_inverse=True)
            # knots=np.quantile(uniquex,[0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
            knots = [np.min(uniquex)]
            knots.extend(np.quantile(uniquex, [0.33, 0.67]))
            knots.extend([np.max(uniquex)])
            KNs[:,i]=knots
            xex[:, i * 4:(i+1) * 4] = truncated_power_basis(xtrain[:, i], knots)#前4列为第一个因子，后4列为第二个因子
        for i in range(0,4):
            xexpand[:,i*4:(i+1)*4]=xex[:,i].reshape((Nlen,1))*xex[:,4:]
    else:
        for i in range(0, xtrain.shape[1]):
            xex[:, i * 4:(i+1) * 4] = truncated_power_basis(xtrain[:, i], KNs[:,i])#前4列为第一个因子，后4列为第二个因子
        for i in range(0,4):
            xexpand[:,i*4:(i+1)*4]=xex[:,i].reshape((Nlen,1))*xex[:,4:]
    return xexpand,KNs


mean1=[0,1]#生成第一类均值所用的均值
mean2=[1,0]#生成第二类均值所用的均值
cov=np.eye(2)#生成均值所用的协方差矩阵
meannum=10#每类的均值个数
x1=np.random.multivariate_normal(mean1,cov,meannum)
x2=np.random.multivariate_normal(mean2,cov,meannum)
covtrain=np.eye(2)/5#生成训练样本的协方差矩阵
covtraininv=np.linalg.inv(covtrain)
trainnum=100
x1train=np.zeros(shape=(trainnum,2))
x2train=np.zeros(shape=(trainnum,2))
for i in range(0,trainnum):
    x1train[i]=np.random.multivariate_normal(x1[np.random.randint(0,meannum),:],covtrain,1)[0]
    x2train[i]=np.random.multivariate_normal(x2[np.random.randint(0, meannum), :], covtrain, 1)[0]
# fig=plt.figure()
# ax1=fig.add_subplot(1,1,1)
# plt.scatter(x1train[:,0],x1train[:,1],marker="o",c="orange")
# ax1.scatter(x2train[:,0],x2train[:,1],marker="o",c="purple")
# plt.show()

#训练样本集
xtrain=np.vstack((x1train,x2train))#两个矩阵按列合并，np.vstack的参数为tuple
ytrain=np.vstack((np.ones(shape=(trainnum,1)),np.zeros(shape=(trainnum,1))))[:,0]

#测试样本集
testnum=10000
xtest=np.zeros(shape=(testnum,2))
ytest=np.zeros(shape=(testnum,))
for i in range(testnum):
    tclass=np.random.randint(0,2)#随机生成一类
    ytest[i]=tclass
    if tclass==1:
        xtest[i]=np.random.multivariate_normal(x1[np.random.randint(0,meannum),:],covtrain,1)[0]
    else:
        xtest[i] = np.random.multivariate_normal(x2[np.random.randint(0, meannum), :], covtrain, 1)[0]

#把二维画布中的每个点分类，以得到boundary curve
paintnum=100
x1paint=np.linspace(np.min(xtrain[:,0]),np.max(xtrain[:,0]),paintnum)
#根据x1的间隔计算x2的间隔，使两个维度间隔相同
x2paint=np.linspace(np.min(xtrain[:,1]),np.max(xtrain[:,1]),
                    (int)(np.round(1.0/((np.max(xtrain[:,0])-
                                         np.min(xtrain[:,0]))/paintnum)*(np.max(xtrain[:,1])-np.min(xtrain[:,1])))))
xx, yy = np.meshgrid(x1paint, x2paint )
xy=np.array([np.array(a) for a in zip(xx.reshape((xx.shape[0]*xx.shape[1],)),yy.reshape((yy.shape[0]*yy.shape[1],)))])

#bayes error rate, sum
bayesclassifysum=np.zeros(shape=(testnum,))#存储贝叶斯分类的结果
for i in range(0,testnum):
    prob = np.zeros(shape=(2, ))  # 计算某一个样本属于各类的概率
    for j in range(meannum):
        prob[0]=prob[0]+np.exp(-(xtest[i,:]-x1[j,:]).dot(covtraininv).dot((xtest[i,:]-x1[j,:]).T)/2)
        prob[1] = prob[1] + np.exp(
            -(xtest[i, :] - x2[j, :]).dot(covtraininv).dot((xtest[i, :] - x2[j, :]).T) / 2)
    if prob[0]>prob[1]:
        bayesclassifysum[i]=1
    else:
        bayesclassifysum[i]=0
errorbayessum=np.count_nonzero(bayesclassifysum!=ytest)/(len(ytest))
print("bayes:"+str(errorbayessum))

Nlen=xtrain.shape[0]
#Additive Natural Cubic Splines - 4 df each

#train error
xexpand,KNs=compute_basis(xtrain)
betaadd,_=lr(xexpand,ytrain)
classifytrain=np.array(xexpand.dot(betaadd)>0)[:,0]
trainerror=np.count_nonzero(classifytrain!=ytrain)/len(ytrain)
print('Additive train error:'+str(trainerror))

#test error
xexpand,_=compute_basis(xtest,KNs)
classifytest=np.array(xexpand.dot(betaadd)>0)[:,0]
testerror=np.count_nonzero(classifytest!=ytest)/len(ytest)
print('Additive test error:'+str(testerror))

#画布
xexpand,_=compute_basis(xy,KNs)
classify=np.array(xexpand.dot(betaadd)>0)[:,0]

fig, ax = plt.subplots()
ax.scatter(xy[classify,0],xy[classify,1],color='b',marker='o',s=0.5)
ax.scatter(xy[~classify,0],xy[~classify,1],color='r',marker='o',s=0.5)

ax.scatter(x1train[:,0],x1train[:,1],marker="o",c="b")
ax.scatter(x2train[:,0],x2train[:,1],marker="o",c="r")

string="bayes:"+str(errorbayessum)+"\n"+'Additive train error:'+str(trainerror)+"\n"+'Additive test error:'+str(testerror)
ax.text(np.min(xtrain[:,0]),np.min(xtrain[:,1]),string,fontsize=10,verticalalignment="bottom",horizontalalignment="left",bbox=dict(facecolor='white', alpha=1))

ax.set_title('Additive Natural Cubic Splines - 4 df each')
#Natural Cubic Splines - Tensor Product - 4 df each
print("bayes:"+str(errorbayessum))
#train error
xexpand,KNs=compute_tensor_basis(xtrain)
betatensor,_=lr(xexpand,ytrain)
classifytrain=np.array(xexpand.dot(betatensor)>0)[:,0]
trainerror=np.count_nonzero(classifytrain!=ytrain)/len(ytrain)
print('Tensor train error:'+str(trainerror))

#test error
xexpand,_=compute_tensor_basis(xtest,KNs)
classifytest=np.array(xexpand.dot(betatensor)>0)[:,0]
testerror=np.count_nonzero(classifytest!=ytest)/len(ytest)
print('Tensor test error:'+str(testerror))

#画布
xexpand,_=compute_tensor_basis(xy,KNs)
classify=np.array(xexpand.dot(betatensor)>0)[:,0]

fig, ax = plt.subplots()
ax.scatter(xy[classify,0],xy[classify,1],color='b',marker='o',s=0.5)
ax.scatter(xy[~classify,0],xy[~classify,1],color='r',marker='o',s=0.5)
ax.scatter(x1train[:,0],x1train[:,1],marker="o",c="b")
ax.scatter(x2train[:,0],x2train[:,1],marker="o",c="r")

string="bayes:"+str(errorbayessum)+"\n"+'Tensor train error:'+str(trainerror)+"\n"+'Tensor test error:'+str(testerror)
ax.text(np.min(xtrain[:,0]),np.min(xtrain[:,1]),string,fontsize=10,verticalalignment="bottom",horizontalalignment="left",bbox=dict(facecolor='white', alpha=1))
ax.set_title('Natural Cubic Splines - Tensor Product - 4 df each')
plt.show()
plt.show()