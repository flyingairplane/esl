##figure 5.4, table 5.1，使用r的ns生成基函数
##2020/09/21
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import seaborn as sns

#logistic regression
def lr(xtrain,ydata):
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
totaldata=pd.read_csv(currentfolder+'SAheart.data.txt',delimiter=',',header=0,index_col=0)

Nlen=totaldata.shape[0]
featurelen=totaldata.shape[1]-1
xdata=totaldata.iloc[:,0:featurelen]
ydata=totaldata.iloc[:,totaldata.shape[1]-1]

xdata=totaldata.loc[:,['sbp','tobacco','ldl','famhist','obesity','age']]

#将二值变量移动到最前面
df_id = xdata.famhist
df = xdata.drop('famhist',axis=1)
df.insert(0,'famhist',df_id)
xdata=df

xexpand=pd.read_csv(currentfolder+'ns.csv',delimiter=',',header=0,index_col=0)

#最终的模型
currentfactors=xdata.columns.values
finalbasis=np.array(xexpand)
beta,weight = lr(finalbasis, ydata)
loglik=np.sum(finalbasis.dot(beta)[:,0]*ydata-np.log(1+np.exp(finalbasis.dot(beta)))[:,0])
finalAIC = -2 * loglik + 2 * finalbasis.shape[1]
finalDev=-2 * loglik
print(finalAIC)

#Fitted natural-spline functions，由于前面的计算中，famhist在模型中，因此假设该变量在模型中
#由于famhist是二值变量，因此不绘图
Eta=np.linalg.inv(finalbasis.T.dot(weight).dot(finalbasis))

fig=plt.figure()
for i in range(1,len(currentfactors)):
    fX = finalbasis[:, (i-1)*4+2:i*4+2].dot(beta[(i-1)*4+2:i*4+2, :])[:, 0]
    temp=np.diag(Eta[(i-1)*4+2:i*4+2, (i-1)*4+2:i*4+2])
    vX = np.sqrt(np.diag(finalbasis[:, (i-1)*4+2:i*4+2].dot(Eta[(i-1)*4+2:i*4+2, (i-1)*4+2:i*4+2]).dot(finalbasis[:, (i-1)*4+2:i*4+2].T)))
    ax = fig.add_subplot(3,2,i)
    sns.lineplot(np.sort(xdata[currentfactors[i]],kind='stable'), fX[np.argsort(xdata[currentfactors[i]],kind='stable')], ax=ax)
    sns.lineplot(np.sort(xdata[currentfactors[i]],kind='stable'), (fX + 2 * vX)[np.argsort(xdata[currentfactors[i]],kind='stable')], ax=ax)
    sns.lineplot(np.sort(xdata[currentfactors[i]],kind='stable'), (fX - 2 * vX)[np.argsort(xdata[currentfactors[i]],kind='stable')], ax=ax)
    sns.rugplot(np.sort(xdata[currentfactors[i]],kind='stable'), color='k')
    ax.set_xlabel(currentfactors[i])
    ax.set_ylabel(r'$\hatf$' + '(' + currentfactors[i] + ')')
plt.show()

#最终模型删除变量后各项指标的变化
famhistin=True
tempbasis=finalbasis
maxAIC=finalAIC
tempAICs=np.zeros((len(currentfactors),))
tempDevs=np.zeros((len(currentfactors),))
# 第一个二值变量
if famhistin==True:
    tempx=tempbasis[:,list(set(range(0,tempbasis.shape[1]))-set([1]))]
    beta,_ = lr(tempx, ydata)
    loglik=np.sum(tempx.dot(beta)[:,0]*ydata-np.log(1+np.exp(tempx.dot(beta)))[:,0])
    tempAICs[0] = -2 * loglik + 2 * (tempbasis.shape[1]-1)
    tempDevs[0]=-2 * loglik
    for i in range(1,len(currentfactors)):
        tempx = tempbasis[:, list(set(range(0,tempbasis.shape[1]))-set(range((i-1)*4+2,i*4+2)))]
        beta,_ = lr(tempx, ydata)
        loglik=np.sum(tempx.dot(beta)[:,0]*ydata-np.log(1+np.exp(tempx.dot(beta)))[:,0])
        tempAICs[i] = -2 * loglik + 2 * (tempbasis.shape[1] - 4)
        tempDevs[i] = -2 * loglik
else:
    for i in range(0,len(currentfactors)):
        tempx = tempbasis[:, list(set(range(0,tempbasis.shape[1]))-set(range(i*4+1,(i+1)*4+1)))]
        beta,_ = lr(tempx, ydata)
        loglik=np.sum(tempx.dot(beta)[:,0]*ydata-np.log(1+np.exp(tempx.dot(beta)))[:,0])
        tempAICs[i] = -2 * loglik + 2 * (tempbasis.shape[1] - 4)
        tempDevs[i] = -2 * loglik

LRT=np.zeros((len(currentfactors),))
pvalues=np.zeros((len(currentfactors),))
for i in range(0,len(currentfactors)):
    LRT[i]=tempDevs[i]-finalDev

# 第一个二值变量
if famhistin==True:
    F=(tempDevs[0]-finalDev)/((finalDev/(Nlen-finalbasis.shape[1])))
    pvalues[0]=stats.f.sf(F,1,Nlen-finalbasis.shape[1])
    for i in range(1,len(currentfactors)):
        F = ((tempDevs[i] - finalDev)/4) / ((finalDev / (Nlen - finalbasis.shape[1])))
        pvalues[i] = stats.f.sf(F, 4, Nlen - finalbasis.shape[1])
else:
    for i in range(0,len(currentfactors)):
        F = ((tempDevs[i] - finalDev) / 4) / ((finalDev / (Nlen - finalbasis.shape[1])))
        pvalues[i] = stats.f.sf(F, 4, Nlen - finalbasis.shape[1])

Df=np.zeros((len(currentfactors)+1,))
Df[0]=np.nan
if famhistin==True:
    Df[1]=1
    for i in range(2,len(currentfactors)+1):
        Df[i]=4
else:
    for i in range(1,len(currentfactors)+1):
        Df[i]=4

output=pd.DataFrame(Df,index=np.concatenate((['none'],currentfactors)),columns=['Df'])
output['Deviance']=np.concatenate(([finalDev],tempDevs))
output['AIC']=np.concatenate(([finalAIC],tempAICs))
output['LRT']=np.concatenate(([np.nan],LRT))
output['P-value']=np.concatenate(([np.nan],pvalues))
writer1=pd.ExcelWriter('table5_1_ns.xlsx')
output.to_excel(writer1,'1')
writer1.save()