###esl第三章table3.1和table3.2
###线性最小二乘以及Zscore
###Date:2020/07/06
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats

currentfolder='D:\\Program\\python\\esl\\Prostate\\'
xlsx = pd.ExcelFile(currentfolder+'ProstateData.xlsx')
totaldata=pd.read_excel(xlsx, sheet_name='Sheet1',header=0,index_col=0)

#train data
traindata=totaldata[totaldata.train=='T']
traindata=traindata.iloc[:,0:9]
txdata=traindata.iloc[:,0:8]
tydata=traindata.iloc[:,8]

# #散点图
# fig=plt.figure(1,figsize=(14,14))
# ax=fig.add_subplot(1,1,1)
# ax=sns.pairplot(data=traindata)
# plt.savefig('scattermatrix1.png')
# plt.show()

# #P50的相关矩阵
# xcorr=txdata.corr()
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax = sns.heatmap(xcorr, annot=True,  fmt='.3f',cmap="BuPu")
# plt.title('correlation coefs')
# plt.show()

#线性最小二乘
Nlen=txdata.shape[0]
featurelen=txdata.shape[1]
#均值为0，方差为1
ss = preprocessing.StandardScaler()
txdata_ss = ss.fit_transform(txdata)
xtrain=np.hstack((np.ones(shape=(len(txdata),1)),txdata_ss))#加一列全1，求beta

#求系数
betalinear=np.linalg.inv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(tydata)
output=pd.DataFrame(betalinear.reshape(9,1),index=np.concatenate((['Intercept'],txdata.columns.values)),columns=['Term'])

ylinear=xtrain.dot(betalinear)
tao=np.sqrt(np.sum(np.square(tydata-ylinear))/(Nlen-featurelen-1))#求拟合标准差
matxtrain=np.array(xtrain)
invXTX=np.linalg.inv(np.dot(matxtrain.T,matxtrain))
diaginv=np.diag(invXTX)
varbeta=tao*np.sqrt(diaginv)
#标准差
output['std error']=varbeta
zscore=betalinear/varbeta
#Z-score
output['zscore']=zscore
print('z scores:',zscore)
writer1=pd.ExcelWriter(currentfolder+'zscores.xlsx')
output.to_excel(writer1,'1')
writer1.save()
#zscore的显著性临界点
significance=stats.t.isf(0.025,Nlen-featurelen-1)
print('critical point:'+str(significance))

#dummy