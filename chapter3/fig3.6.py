###esl第三章fig3.6
###best subset,forward stepwise,backward stepwise, forward stagewise
###Date:2020/07/07未完成
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
p=31#因子数量
N=300#训练样本数量
validnum=10#有效因子数量
cov=np.ones(p)*0.85+np.eye(p)*0.15
xtrain=np.random.multivariate_normal(np.zeros(shape=(1,p))[0],cov,N)
tencoefs=np.random.normal(0,np.sqrt(0.4),size=(validnum,1))
coefs=np.vstack((tencoefs,np.zeros(shape=(p-validnum,1))))
noise=np.random.normal(0,np.sqrt(6.25),size=(N,p))
ytrain=xtrain.dot(coefs)+noise

#Forward stepwise
predictorset=[]
#for i in range(1,p+1):#因子集内因子数量
