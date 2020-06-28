###fig24
###求optimal bayes error rate
##sum指的一类中的所有概率相加
##max指的一类中的最大概率
##仿真结果表明sum的错误率更低
# ###date:2020/06/26
import numpy as np
import matplotlib.pyplot as plt
mean1=[0,1.1]#生成第一类均值所用的均值
mean2=[1.1,0]#生成第二类均值所用的均值
cov=np.eye(2)#生成均值所用的协方差矩阵
meannum=10#每类的均值个数
x1=np.random.multivariate_normal(mean1,cov,meannum)
x2=np.random.multivariate_normal(mean2,cov,meannum)
covtrain=np.eye(2)/5#生成训练样本的协方差矩阵
covtraininv=np.linalg.inv(covtrain)
simnum=100#训练样本集数量
trainnum=1000#训练样本集中一类的样本数
errorbayesmax=np.zeros(shape=(simnum,1))
errorbayessum=np.zeros(shape=(simnum,1))
for simk in range(simnum):
    x1train=np.zeros(shape=(trainnum,2))
    x2train=np.zeros(shape=(trainnum,2))
    for i in range(0,trainnum):
        x1train[i]=np.random.multivariate_normal(x1[np.random.randint(0,meannum),:],covtrain,1)[0]
        x2train[i]=np.random.multivariate_normal(x2[np.random.randint(0, meannum), :], covtrain, 1)[0]

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
    errorbayessum[simk]=np.count_nonzero(bayesclassifysum!=ytrain)/(len(ytrain))
    #print("求和bayes:"+str(errorbayessum[simk]))

    #bayes error rate, max
    bayesclassifymax=np.zeros(shape=(trainnum*2,1))#存储贝叶斯分类的结果
    for i in range(0,trainnum*2):
        prob = np.zeros(shape=(meannum*2, 1))  # 计算某一个样本属于各分布的概率
        for j in range(meannum):
            prob[j]=np.exp(-(xtrainorigin[i,:]-x1[j,:]).dot(covtraininv).dot((xtrainorigin[i,:]-x1[j,:]).T)/2)
            prob[j+meannum] =np.exp(-(xtrainorigin[i, :] - x2[j, :]).dot(covtraininv).dot((xtrainorigin[i, :] - x2[j, :]).T) / 2)
        bayesclassifymax[i]=1 if prob.argmax()<meannum else 0
    errorbayesmax[simk]=np.count_nonzero(bayesclassifymax!=ytrain)/(len(ytrain))
    #print("最大bayes:"+str(errorbayesmax[simk]))
# fig=plt.figure()
# ax2=fig.add_subplot(1,1,1)
# ax2.plot(range(simnum),errorbayessum,'bo-',label='bayes sum')
# ax2.plot(range(simnum),errorbayesmax,'ro-',label='bayes max')
# plt.legend(loc='upper right')
# plt.show()

fig=plt.figure()
ax2=fig.add_subplot(1,1,1)
ax2.plot(range(simnum),errorbayesmax-errorbayessum,'bo-',label='bayes delta')
plt.legend(loc='upper right')
plt.show()