import numpy as np
class treeSearch:
    def __init__(self,tree,fold,x,y):
        self.tree=tree
        self.mseup=10000
        self.minmse=self.mseup#对于指定深度，计算的最小mse
        self.se=0#对于指定深度，最小mse对应的standard error
        self.fold=fold

        self.x=x
        self.y=y

        self.bscverror = np.zeros(shape=(x.shape[1],))  # 每个因子数量对应的平均cv error
        self.bsse = np.zeros(shape=(x.shape[1],))  # 每个因子数量对应的standard error

        self.trainindex,self.testindex=self.cvsplit(x.shape[0],fold)

    def mseinitilize(self):
        self.minmse = self.mseup

    def cvsplit(self,n,k):#n为训练集样本数,k为k-fold
        testX=[]
        trainX=[]
        eachnum=round(n/k)
        foldnum=eachnum*np.ones((k,))
        if eachnum*k>n:
            index = np.sort(np.random.choice(np.arange(0, k), replace=False, size=eachnum*k-n))
            foldnum[index]=foldnum[index]-1
        elif eachnum*k<n:
            index = np.sort(np.random.choice(np.arange(0, k), replace=False, size=n-eachnum * k))
            foldnum[index] = foldnum[index] + 1
        for i in range(0,k):
            index=list(np.arange(np.sum(foldnum[0:i]),np.sum(foldnum[0:i])+foldnum[i]))
            index=[int(i) for i in index]
            testX.append(index)
            trainX.append(list(set(np.arange(0, n)) - set(index)))
        return trainX,testX

    def countMSE(self,indexes):
        mses=np.zeros((self.fold,))
        for i in range(0,self.fold):
            txtrain = self.x[self.trainindex[i], :]
            txtrain = txtrain[:, [int(i) for i in np.hstack(([0], indexes))]]
            txtest=self.x[self.testindex[i], :]
            txtest = txtest[:, [int(i) for i in np.hstack(([0], indexes))]]
            betahat = np.linalg.inv(txtrain.T.dot(txtrain)).dot(txtrain.T).dot(self.y.iloc[self.trainindex[i]])
            typredict = txtest.dot(betahat)
            mses[i]=np.mean(np.square(self.y.iloc[self.testindex[i]] - typredict))

        return np.mean(mses), np.std(mses)/np.sqrt(self.fold)
    #tn为父结点
    # depth为因子集的因子数量
    def search(self,tn, depth):
        if depth==0:
            self.minmse,self.se=self.countMSE([])
            return

        if tn.mse == -1:  # 说明本节点未计算过MSE
            tn.mse, se = self.countMSE(tn.indexes)
        if len(tn.indexes) == depth:
            if self.minmse == self.mseup or self.minmse>tn.mse:  # 首次计算MSE或者MSE变小，更新
                self.minmse = tn.mse
                self.se=se
        else:
            if self.minmse == self.mseup or self.minmse > tn.mse:  # 如果该节点对应的MSE高于前期的minmse，则不再递归，利用了三角不等式
                for child in tn.children:
                    self.search(child, depth)
    #bscverror为best subset的cross-validation MSE
    #bsse为best subset的standard error
    def dosearch(self):
        for i in range(len(self.bscverror) - 1, -1, -1):  # 因子数量
        #for i in range(0, -1, -1):  # 因子数量
            print('number of predictors:'+str(i))
            self.mseinitilize()
            self.search(self.tree,i)
            self.bscverror[i]=self.minmse
            self.bsse[i]=self.se
        return self.bscverror,self.bsse





