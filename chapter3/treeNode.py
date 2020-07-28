#leaps and bounds算法，因子数量为31时超内存了
class TreeNode:
    def __init__(self,indexes,last=-1):
        self.last=last#上一个删除的predictor,默认为根节点
        self.indexes=indexes#节点包含的predictor，为数组
        self.children=[]
        self.mse=-1
        if len(indexes)>1:
            for index in indexes:
                if index>last:
                    temp=indexes.copy()
                    temp.remove(index)
                    self.children.append(TreeNode(temp,index))

    def initilize(self):
        self.mse=-1
        for child in self.children:
            child.initilize()



if __name__=="__main__":
    featurelen=8
    a=TreeNode([i for i in range(1,featurelen+1)])



