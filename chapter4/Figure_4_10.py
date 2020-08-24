##figure 4.8/4.10/4.11
##2020/08/14
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
currentfolder='D:\\Program\\python\\esl\\chapter4\\'
traindata=pd.read_csv(currentfolder+'vowel.train.txt',delimiter=',',header=0,index_col=0)
testdata=pd.read_csv(currentfolder+'vowel.test.txt',delimiter=',',header=0,index_col=0)

Nlen=traindata.shape[0]
featurelen=traindata.shape[1]-1
x_train=traindata.iloc[:,1:]
y_train=traindata.iloc[:,0]
classnum=len(set(y_train))

x_test=testdata.iloc[:,1:]
y_test=testdata.iloc[:,0]

PaiK=np.zeros(shape=(classnum,))#每类的先验
MiuK=np.zeros(shape=(classnum,featurelen))#每类的均值
EtaK=np.zeros(shape=(classnum,featurelen,featurelen))#每类的协方差
EtaT=np.zeros(shape=(featurelen,featurelen))#类内协方差
for i in range(0,classnum):
    subnum=np.count_nonzero(y_train==i+1)
    PaiK[i] = subnum / Nlen
    subxdata=x_train.loc[y_train==i+1,:]
    submiu=np.mean(subxdata,axis=0)
    MiuK[i,:]=submiu
    subeta=((subxdata-submiu).T).dot(subxdata-submiu)
    EtaK[i,:,:]=subeta/(subnum-1)
    EtaT=EtaT+subeta
EtaT=EtaT/(Nlen-classnum)
Miu=np.mean(MiuK,axis=0)
#类间协方差
EtaB=((MiuK-Miu).T).dot(MiuK-Miu)/(classnum-1)

values,vectors=np.linalg.eig(np.linalg.inv(EtaT).dot(EtaB))
ProjectionX_train=x_train.dot(vectors)
ProjectionX_test=x_test.dot(vectors)


##############测试不同维度下的分类错误率，分别使用了QDA和LDA
dims=np.arange(1,featurelen+1)#维度
errortrain=np.zeros((len(dims),))
errortest=np.zeros((len(dims),))
traindelta=np.zeros((len(y_train),classnum))
testdelta=np.zeros((len(y_test),classnum))
for k in range(0,len(dims)):
    dim=dims[k]
    print(dim)

    miuk = np.zeros(shape=(classnum, dim))  # 每类的均值
    etak = np.zeros(shape=(classnum, dim, dim))  # 每类的协方差
    etat = np.zeros(shape=(dim, dim))  # 总的协方差
    invetak = np.zeros(shape=(classnum, dim, dim))  # 每类的协方差求逆
    detetak = np.zeros(shape=(classnum,))  # 每类协方差的行列式


    pX_train=ProjectionX_train.iloc[:,0:k+1]
    pX_test = ProjectionX_test.iloc[:, 0:k+1]
    for i in range(0, classnum):
        subnum = np.count_nonzero(y_train == i + 1)
        subxdata = pX_train.loc[y_train == i + 1, :]
        submiu = np.mean(subxdata, axis=0)
        miuk[i, :] = submiu
        subeta = ((subxdata - submiu).T).dot(subxdata - submiu)
        etak[i, :, :] = subeta / (subnum - 1)
        etat = etat + subeta
    etat = etat / (Nlen - classnum)
    miu = np.mean(miuk, axis=0)

    # #qda
    # for j in range(0,classnum):
    #     subeta=etak[j,:,:]
    #     invetak[j,:,:]=np.linalg.inv(subeta)
    #     detetak[j]=np.linalg.det(subeta)
    # for i in range(0,len(y_train)):
    #     for j in range(0,classnum):
    #         traindelta[i, j] =-np.log(detetak[j])/2- (pX_train.iloc[i, :]-miuk[j, :]).dot(invetak[j,:,:]).dot((pX_train.iloc[i, :]-miuk[j, :]).T)/2 + np.log(PaiK[j])
    #
    # for i in range(0,len(y_test)):
    #     for j in range(0,classnum):
    #         testdelta[i, j] =-np.log(detetak[j])/2- (pX_test.iloc[i, :]-miuk[j, :]).dot(invetak[j,:,:]).dot((pX_test.iloc[i, :]-miuk[j, :]).T)/2 + np.log(PaiK[j])
    # errortrain[k]=np.count_nonzero(y_train!=(np.argmax(traindelta,axis=1)+1))/len(y_train)
    # errortest[k] = np.count_nonzero(y_test != (np.argmax(testdelta, axis=1) + 1)) / len(y_test)

    #lda
    invetat=np.linalg.inv(etat)  # 每类的协方差求逆
    for i in range(0,len(y_train)):
        for j in range(0,classnum):
            traindelta[i, j] =-(pX_train.iloc[i, :]-miuk[j, :]).dot(invetat).dot((pX_train.iloc[i, :]-miuk[j, :]).T)/2 + np.log(PaiK[j])

    for i in range(0,len(y_test)):
        for j in range(0,classnum):
            testdelta[i, j] =-(pX_test.iloc[i, :]-miuk[j, :]).dot(invetat).dot((pX_test.iloc[i, :]-miuk[j, :]).T)/2 + np.log(PaiK[j])
    errortrain[k]=np.count_nonzero(y_train!=(np.argmax(traindelta,axis=1)+1))/len(y_train)
    errortest[k] = np.count_nonzero(y_test != (np.argmax(testdelta, axis=1) + 1)) / len(y_test)

plt.style.use('ggplot')
fig,ax=plt.subplots()
ax.plot(dims,errortrain,'-.',label='train')
ax.plot(dims,errortest,'-.',label='test')
ax.set_ylabel('error rate')
ax.set_xlabel('dimension')
ax.set_title('LDA and Dimension Reduction')
ax.legend(loc='center right')
plt.show()
###############


# ################前两个维度的可视化,类似地可以实现任意两个维度的可视化
# coor1=ProjectionX_train.iloc[:,0]
# coor2=ProjectionX_train.iloc[:,1]
#
# plt.style.use('ggplot')
# fig,ax=plt.subplots()
# for i in range(0,classnum):
#     ax.plot(coor1[y_train==i+1],coor2[y_train==i+1],'o')
# ax.set_ylabel('Coordinate 2')
# ax.set_xlabel('Coordinate 1')
# ax.set_title('Classification in Reduced Subspace')
# plt.show()
########################
