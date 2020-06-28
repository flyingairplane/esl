###fig24
###循环
###date:2020/06/26
import numpy as np
import matplotlib.pyplot as plt
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
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.scatter(x1train[:,0],x1train[:,1],marker="o",c="orange")
ax1.scatter(x2train[:,0],x2train[:,1],marker="o",c="purple")
plt.show()

#训练样本集
xtrainorigin=np.vstack((x1train,x2train))#两个矩阵按列合并，np.vstack的参数为tuple
ytrain=np.vstack((np.ones(shape=(trainnum,1)),np.zeros(shape=(trainnum,1))))

#测试样本集
testnum=10000
xtestorigin=np.zeros(shape=(testnum,2))
ytest=np.zeros(shape=(testnum,1))
for i in range(testnum):
    tclass=np.random.randint(0,2)#随机生成一类
    ytest[i]=tclass
    if tclass==1:
        xtestorigin[i]=np.random.multivariate_normal(x1[np.random.randint(0,meannum),:],covtrain,1)[0]
    else:
        xtestorigin[i] = np.random.multivariate_normal(x2[np.random.randint(0, meannum), :], covtrain, 1)[0]

#bayes error rate
bayesclassify=np.zeros(shape=(testnum,1))#存储贝叶斯分类的结果
for i in range(0,testnum):
    prob = np.zeros(shape=(2, 1))  # 计算某一个样本属于各类的概率
    for j in range(meannum):
        prob[0]=prob[0]+np.exp(-(xtestorigin[i,:]-x1[j,:]).dot(covtraininv).dot((xtestorigin[i,:]-x1[j,:]).T)/2)
        prob[1] = prob[1] + np.exp(
            -(xtestorigin[i, :] - x2[j, :]).dot(covtraininv).dot((xtestorigin[i, :] - x2[j, :]).T) / 2)
    if prob[0]>prob[1]:
        bayesclassify[i]=1
    else:
        bayesclassify[i]=0
errorbayes=np.count_nonzero(bayesclassify!=ytest)/(len(ytest))
print("贝叶斯误差:"+str(errorbayes))

###线性回归
xtrain=np.hstack((np.ones(shape=(len(xtrainorigin),1)),xtrainorigin))#加一列全1，求beta0
xtest=np.hstack((np.ones(shape=(len(xtestorigin),1)),xtestorigin))#加一列全1
betalinear=np.linalg.inv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)

#训练误差
ylinear=xtrain.dot(betalinear)
yhatlineartrain=np.where(ylinear>0.5,1,0)
errorlineartrain=np.count_nonzero(yhatlineartrain!=ytrain)/(len(ytrain))
print("线性回归训练误差:"+str(errorlineartrain))

ylineartest=xtest.dot(betalinear)
yhatlineartest=np.where(ylineartest>0.5,1,0)
errorlineartest=np.count_nonzero(yhatlineartest!=ytest)/(len(ytest))
print("线性回归测试误差:"+str(errorlineartest))

#kNN
knum=range(1,int(trainnum*2*0.75),2)#最多用到75%的近邻
#knum=[1,3,5,7,11,21,31,45,69,101,151]
distancetrain=np.zeros(shape=(trainnum*2,trainnum*2))
for i in range(len(xtrainorigin)):
    for j in range(i+1,len(xtrainorigin)):
        distancetrain[i,j]=np.sqrt(sum((xtrainorigin[i,:]-xtrainorigin[j,:])**2))
        distancetrain[j,i]=distancetrain[i,j]
distancetest=np.zeros(shape=(testnum,trainnum*2))
for i in range(len(xtestorigin)):
    for j in range(0,trainnum*2):
        distancetest[i,j]=np.sqrt(sum((xtestorigin[i,:]-xtrainorigin[j,:])**2))
classifytrain=np.zeros(shape=(trainnum*2,1))#训练样本集中每个样本的分类结果
classifytest=np.zeros(shape=(testnum,1))#测试样本集中每个样本的分类结果
errorknntrain=np.zeros(shape=(len(knum),1))
errorknntest=np.zeros(shape=(len(knum),1))
for k in range(len(knum)):
    #训练误差
    for i in range(len(xtrainorigin)):
        classifytrain[i]=1 if np.mean(ytrain[np.argsort(distancetrain[i,:])[0:knum[k]]])>0.5 else 0
    errorknntrain[k]=np.count_nonzero(classifytrain!=ytrain)/(len(ytrain))
    #测试误差
    for i in range(len(xtestorigin)):
        classifytest[i]=1 if np.mean(ytrain[np.argsort(distancetest[i,:])[0:knum[k]]])>0.5 else 0
    errorknntest[k]=np.count_nonzero(classifytest!=ytest)/(len(ytest))

fig1=plt.figure()
ax2=fig1.add_subplot(1,1,1)
ax2.plot(knum,errorknntrain,'bo-',label='knn train')
ax2.plot(knum,errorknntest,'ro-',label='knn test')
ax2.plot(knum,errorbayes*np.ones(shape=(len(knum),1)),'k-',label='bayes')
ax2.plot(trainnum*2/3,errorlineartrain,'bs',label='linear train')#3为线性回归的自由度
ax2.plot(trainnum*2/3,errorlineartest,'rs',label='linear test')
plt.legend(loc='upper right')
plt.show()