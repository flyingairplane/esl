###fig24
###单次
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
plt.scatter(x1train[:,0],x1train[:,1],marker="o",c="orange")
ax1.scatter(x2train[:,0],x2train[:,1],marker="o",c="purple")
plt.show()

#训练样本集
xtrainorigin=np.vstack((x1train,x2train))#两个矩阵按列合并，np.vstack的参数为tuple
ytrain=np.vstack((np.ones(shape=(trainnum,1)),np.zeros(shape=(trainnum,1))))

#bayes error rate, sum
bayesclassifysum=np.zeros(shape=(trainnum*2,1))#存储贝叶斯分类的结果
for i in range(0,trainnum*2):
    prob = np.zeros(shape=(2, 1))  # 计算某一个样本属于各类的概率
    for j in range(meannum):
        prob[0]=prob[0]+np.exp(-(xtrainorigin[i,:]-x1[j,:]).dot(covtraininv).dot((xtrainorigin[i,:]-x1[j,:]).T)/2)
        prob[1] = prob[1] + np.exp(
            -(xtrainorigin[i, :] - x2[j, :]).dot(covtraininv).dot((xtrainorigin[i, :] - x2[j, :]).T) / 2)
    if prob[0]>prob[1]:
        bayesclassifysum[i]=1
    else:
        bayesclassifysum[i]=0
errorbayessum=np.count_nonzero(bayesclassifysum!=ytrain)/(len(ytrain))
print("求和bayes:"+str(errorbayessum))

# #bayes error rate, max。似乎误差高于sum
# bayesclassifymax=np.zeros(shape=(trainnum*2,1))#存储贝叶斯分类的结果
# for i in range(0,trainnum*2):
#     prob = np.zeros(shape=(meannum*2, 1))  # 计算某一个样本属于各分布的概率
#     for j in range(meannum):
#         prob[j]=np.exp(-(xtrainorigin[i,:]-x1[j,:]).dot(covtraininv).dot((xtrainorigin[i,:]-x1[j,:]).T)/2)
#         prob[j+meannum] =np.exp(-(xtrainorigin[i, :] - x2[j, :]).dot(covtraininv).dot((xtrainorigin[i, :] - x2[j, :]).T) / 2)
#     bayesclassifymax[i]=1 if prob.argmax()<meannum else 0
# errorbayesmax=np.count_nonzero(bayesclassifymax!=ytrain)/(len(ytrain))
# print("最大bayes:"+str(errorbayesmax))

#线性回归
xtrain=np.hstack((np.ones(shape=(len(xtrainorigin),1)),xtrainorigin))#加一列全1，求beta0
betalinear=np.linalg.inv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)
ylinear=xtrain.dot(betalinear)
yhatlinear=np.where(ylinear>0.5,1,0)
errorlinear=np.count_nonzero(yhatlinear!=ytrain)/(len(ytrain))
print(errorlinear)

#kNN
distance=np.zeros(shape=(trainnum*2,trainnum*2))
for i in range(len(xtrainorigin)):
    for j in range(i+1,len(xtrainorigin)):
        distance[i,j]=np.sqrt(sum((xtrainorigin[i,:]-xtrainorigin[j,:])**2))
        distance[j,i]=distance[i,j]
classify=np.zeros(shape=(trainnum*2,1))
k=1
for i in range(len(xtrainorigin)):
    classify[i]=1 if np.mean(ytrain[np.argsort(distance[i,:])[:k]])>0.5 else 0

errorknn=np.count_nonzero(classify!=ytrain)/(len(ytrain))
print(errorknn)