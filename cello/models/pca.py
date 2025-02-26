# 导入必要的库
from optparse import OptionParser
import sklearn
from sklearn import decomposition
import dill

# 主函数定义
def main():
    # TODO: 待添加使用说明
    # usage = "" # TODO 
    # parser = OptionParser(usage=usage)
    # #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    # #parser.add_option("-b", "--b_descrip", help="This is an argument")
    # (options, args) = parser.parse_args()

    # 创建测试数据：4个5维向量
    vecs = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 2.0, 5.0, 14.0, 25.0],
        [133.0, 2.0, 35.0, 4.0, 35.0],
        [12.0, 2.0, 32.0, 4.0, 15.0],
    ]
    # 创建PCA模型并测试（降至2维）
    model = PCA(params={'n_components':4})
    model.fit(vecs)
    print(model.transform(vecs))

# PCA降维类定义
class PCA:
    def __init__(self, params):
        """
        初始化PCA类
        params: 包含PCA参数的字典，必须包含'n_components'键
        """
        # 获取目标维度
        n_dims = params['n_components']
        self.n_dims = n_dims
        # 创建PCA模型，使用随机SVD求解器
        self.model = decomposition.PCA(n_dims, svd_solver='randomized')

    # 拟合方法：学习数据的主成分
    def fit(self, X):
        print('Fitting PCA with {} components...'.format(self.n_dims))
        self.model.fit(X)
        print('done.')

    # 属性：获取主成分矩阵
    @property
    def components_(self):
        return self.model.components_

    # 转换方法：将数据投影到主成分空间
    def transform(self, X):
        print('Transforming with PCA...')
        X = self.model.transform(X)
        print('done.')
        return X

# 程序入口点
if __name__ == "__main__":
    main()
