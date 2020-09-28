##figure 5.6
##2020/09/23
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#参考维基百科https://en.wikipedia.org/wiki/Smoothing_spline#cite_note-Rodriguez-6
# 中的“Derivation of the cubic smoothing spline”部分
def constructA(knots):
    num=len(knots)
    delta=np.zeros((num-2,num))
    W=np.zeros((num-2,num-2))
    h=np.diff(knots)
    for i in range(0,num-2):
        delta[i,i]=1.0/h[i]
        delta[i,i+1]=-1.0/h[i]-1/h[i+1]
        delta[i,i+2]=1.0/h[i+1]
    for i in range(0,num-2):
        W[i,i]=(h[i]+h[i+1])/3
    for i in range(1,num-2):
        W[i-1,i]=h[i]/6
        W[i,i-1]=h[i]/6
    return delta,W

#一个简单的求lambda的函数
def compute_lambda(K,df):
    lam=0
    step=0.1
    lastsign=1
    sign=1
    tol=1e-4
    Slam=np.trace(np.linalg.inv(np.eye(K.shape[0])))
    while np.abs(Slam-df)>tol:
        if Slam>df:
            sign=1
            if sign==lastsign:
                lam=lam+step
            else:
                step=step/2
                lam=lam+step
            lastsign=sign
        else:
            sign=-1
            if sign!=lastsign:
                step=step/2
                lam=lam-step
            else:
                lam=lam-step
            lastsign=sign
        Slam = np.trace(np.linalg.inv(np.eye(K.shape[0]) + lam * K))
    return lam

currentfolder='D:\\Program\\python\\esl\\chapter5\\'
totaldata=pd.read_csv(currentfolder+'bone.data.txt',delimiter='\t',header=0,index_col=0)
male=totaldata[totaldata.gender=="male"]
female=totaldata[totaldata.gender=="female"]
del male['gender']
del female['gender']

#male
ageunique,counts=np.unique(male.age,return_counts=True)
ageunique.sort()
#重复年龄加权
weights=np.diag(counts[np.argsort(ageunique)])
yunique=[np.mean(male.spnbmd[male.age==age]) for age in ageunique]

df=12
knots=ageunique
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
#拟合结果
f=np.linalg.inv(np.eye(K.shape[0]) + lam * K).dot(np.array(yunique).reshape((len(yunique),1)))
#加权拟合结果
fw=np.linalg.inv(weights + lam * K).dot(weights).dot(np.array(yunique).reshape((len(yunique),1)))

#female
ageuniquefe,countsfe=np.unique(female.age,return_counts=True)
ageuniquefe.sort()
#重复年龄加权
weights=np.diag(countsfe[np.argsort(ageuniquefe)])
yunique=[np.mean(female.spnbmd[female.age==age]) for age in ageuniquefe]

df=12
knots=ageuniquefe
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
#拟合结果
fe=np.linalg.inv(np.eye(K.shape[0]) + lam * K).dot(np.array(yunique).reshape((len(yunique),1)))
#加权拟合结果
fwe=np.linalg.inv(weights + lam * K).dot(weights).dot(np.array(yunique).reshape((len(yunique),1)))

fig, ax = plt.subplots()
ax.scatter(male.age,male.spnbmd,marker='o',s=10)
ax.plot(ageunique,f[:,0],color='b',label='male')
ax.plot(ageunique,fw[:,0],color='b',linestyle='-.',label='male-weight')
ax.scatter(female.age,female.spnbmd,marker='o',s=10)
ax.plot(ageuniquefe,fe[:,0],color='r',label='female')
ax.plot(ageuniquefe,fwe[:,0],color='r',linestyle='-.',label='female-weight')
ax.set_xlabel('age')
ax.set_ylabel('Relative Change in Spinal BMD')
ax.legend(loc='upper right')
plt.show()
