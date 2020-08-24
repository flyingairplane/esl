##figure Table 4.2
##2020/08/20
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
currentfolder='D:\\Program\\python\\esl\\chapter4\\'
totaldata=pd.read_csv(currentfolder+'SAheart.data.txt',delimiter=',',header=0,index_col=0)

xdata=totaldata.loc[:,['sbp','tobacco','ldl','famhist','obesity','alcohol','age']]

Nlen=totaldata.shape[0]
featurelen=xdata.shape[1]

ydata=totaldata.iloc[:,totaldata.shape[1]-1]

xtrain=np.hstack((np.ones(shape=(len(xdata),1)),xdata))#加一列全1
beta_old=np.ones((featurelen+1,1))
beta_new=np.zeros((featurelen+1,1))
tol=1e-4
iternum=1000#最大迭代次数
iter=0
W=np.zeros((featurelen+1,featurelen+1))
while np.sum(np.square(beta_new-beta_old))>tol and iter<iternum:
    iter=iter+1
    print(iter)
    beta_old=beta_new
    p=np.exp(xtrain.dot(beta_old))/(1+np.exp(xtrain.dot(beta_old)))
    W=np.diag((p*(1-p))[:,0])
    z=xtrain.dot(beta_old)+np.diag(1/((p*(1-p))[:,0])).dot(np.array((ydata-p[:,0])).reshape((Nlen,1)))
    beta_new=np.linalg.inv(xtrain.T.dot(W).dot(xtrain)).dot(xtrain.T).dot(W).dot(z)
if iter==iternum:
    print('not convergence.')


stds=np.sqrt(np.diag(np.linalg.inv(xtrain.T.dot(W).dot(xtrain))))
zscore=beta_new[:,0]/stds
output=pd.DataFrame(beta_new,index=np.concatenate((['Intercept'],xdata.columns.values)),columns=['Term'])
output['std error']=stds
output['zscore']=zscore
writer1=pd.ExcelWriter(currentfolder+'zscores.xlsx')
output.to_excel(writer1,'1')
writer1.save()
#zscore的显著性临界点
significance=stats.t.isf(0.025,Nlen-featurelen-1)
print('critical point:'+str(significance))
