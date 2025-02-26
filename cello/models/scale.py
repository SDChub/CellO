# 导入必要的库
import sklearn
from sklearn import preprocessing

# 主函数定义
def main():
    # # TODO: 待添加使用说明
    # usage = "" # TODO 
    # parser = OptionParser(usage=usage)
    # #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    # #parser.add_option("-b", "--b_descrip", help="This is an argument")
    # (options, args) = parser.parse_args()

    # 测试数据：创建示例向量
    vecs = [
        [1.0, 1.0, 3.0, 4.0, 5.0],
        [10.0, 0.0, 5.0, 14.0, 25.0],
        [133.0, 0.0, 1.0, 35.0, 35.0],
        [12.0, 1.0, 32.0, 4.0, 15.0],
    ]

    # 创建并测试Scale模型
    model = Scale(params={'with_std': True})
    model.fit(vecs)
    print(model.transform(vecs))

# 数据标准化类定义
class Scale:
    def __init__(self, params):
        """
        初始化Scale类
        params: 包含标准化参数的字典
        """
        # 保存参数
        self.params = params
        # 获取是否进行标准差缩放的参数
        with_std = params['with_std']
        # 创建StandardScaler实例
        self.model = preprocessing.StandardScaler(
            with_std=with_std
        )

    # 拟合方法：学习数据的均值和标准差
    def fit(self, X):
        print('Fitting scale preprocessor...')
        self.model.fit(X)
        print('done.')

    # 转换方法：使用学习到的参数标准化数据
    def transform(self, X):
        print('Scaling data...')
        X_scaled = self.model.transform(X)
        print('done.')
        return X_scaled

# 程序入口点
if __name__ == "__main__":
    main()
