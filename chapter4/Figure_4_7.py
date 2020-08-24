##figure 4.7
##2020/08/12
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
EtaT=np.zeros(shape=(featurelen,featurelen))#总的协方差
for i in range(0,classnum):
    subnum=np.count_nonzero(y_train==i+1)
    PaiK[i]=subnum/Nlen
    subxdata=x_train.loc[y_train==i+1,:]
    submiu=np.mean(subxdata,axis=0)
    MiuK[i,:]=submiu
    subeta=((subxdata-submiu).T).dot(subxdata-submiu)
    EtaK[i,:,:]=subeta/(subnum-1)
    EtaT=EtaT+subeta
EtaT=EtaT/(Nlen-classnum)

#RDA
traindelta=np.zeros((len(y_train),classnum))
testdelta=np.zeros((len(y_test),classnum))

alphas=np.linspace(0,1,20)
errortrain=np.zeros((len(alphas),))
errortest=np.zeros((len(alphas),))
invEtaK=np.zeros(shape=(classnum,featurelen,featurelen))#每类的协方差求逆
detEtaK=np.zeros(shape=(classnum,))#每类协方差的行列式
for k in range(0,len(alphas)):
    alpha=alphas[k]
    print(alpha)
    for j in range(0,classnum):
        subeta=alpha*EtaK[j,:,:]+(1-alpha)*EtaT
        invEtaK[j,:,:]=np.linalg.inv(subeta)
        detEtaK[j]=np.linalg.det(subeta)

    for i in range(0,len(y_train)):
        for j in range(0,classnum):
            traindelta[i, j] =-np.log(detEtaK[j])/2- (x_train.iloc[i, :]-MiuK[j, :]).dot(invEtaK[j,:,:]).dot((x_train.iloc[i, :]-MiuK[j, :]).T)/2 + np.log(PaiK[j])

    for i in range(0,len(y_test)):
        for j in range(0,classnum):
            testdelta[i, j] =-np.log(detEtaK[j])/2- (x_test.iloc[i, :]-MiuK[j, :]).dot(invEtaK[j,:,:]).dot((x_test.iloc[i, :]-MiuK[j, :]).T)/2 + np.log(PaiK[j])
    errortrain[k]=np.count_nonzero(y_train!=(np.argmax(traindelta,axis=1)+1))/len(y_train)
    errortest[k] = np.count_nonzero(y_test != (np.argmax(testdelta, axis=1) + 1)) / len(y_test)

plt.style.use('ggplot')
fig,ax=plt.subplots()
ax.plot(alphas,errortrain,'-.',label='train')
ax.plot(alphas,errortest,'-.',label='test')
ax.set_ylabel('error rate')
ax.set_xlabel('alpha')
ax.set_title('Regularized Discriminant Analysis')
ax.legend(loc='center right')
plt.show()