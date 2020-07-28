###esl第三章fig3.7
###best subset
###Date:2020/07/07
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import treeNode
import searchTree

currentfolder='D:\\Program\\python\\esl\\Prostate\\'
xlsx = pd.ExcelFile(currentfolder+'ProstateData.xlsx')
totaldata=pd.read_excel(xlsx, sheet_name='Sheet1',header=0,index_col=0)

#train data
traindata=totaldata[totaldata.train=='T']
traindata=traindata.iloc[:,0:9]
txdata=traindata.iloc[:,0:8]
tydata=traindata.iloc[:,8]

Nlen=txdata.shape[0]
featurelen=txdata.shape[1]
fold=10#tenfold cv

#均值为0，方差为1
ss = preprocessing.StandardScaler()
txdata_ss = ss.fit_transform(txdata)
xtrain=np.hstack((np.ones(shape=(len(txdata),1)),txdata_ss))#加一列全1，求beta

#best subset
tree=treeNode.TreeNode([i for i in range(1,featurelen+1)])
tS=searchTree.treeSearch(tree,fold,xtrain,tydata)
# bscverror每个因子数量对应的平均cv error
# bsse每个因子数量对应的standard error
bscverror,bsse=tS.dosearch()

plt.style.use('ggplot')
fig, ax = plt.subplots()
lw=2
plt.errorbar(np.arange(0,len(bscverror)), bscverror, yerr = bsse, fmt='o',ls='-', capsize=4, capthick=2)
plt.plot(np.linspace(0,len(bscverror),100),np.ones(shape=(100,1))*np.min(bscverror+bsse),'-.')
plt.xlabel('Subset Size')
plt.ylabel('CV Error')
plt.title('All Subsets')
plt.show()



