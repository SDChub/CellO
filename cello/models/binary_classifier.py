# 导入必要的机器学习库
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.metrics import average_precision_score
#import random

# L2正则化逻辑回归分类器类
class L2LogisticRegression():
    # 初始化分类器，设置模型参数
    def __init__(self, params):
        solver = params['solver']
        penalty_weight = params['penalty_weight']
        # 根据参数决定是否使用类别权重平衡
        if params['downweight_by_class']:
            class_weight = 'balanced'
        else:
            class_weight = None
        # 如果使用liblinear求解器，设置相应参数
        if solver == 'liblinear':
            intercept_scaling = params['intercept_scaling']
            self.model = LogisticRegression(
                C=penalty_weight,          # 正则化强度的倒数
                penalty='l2',              # 使用L2正则化
                solver='liblinear',        # 使用liblinear求解器
                tol=1e-9,                 # 设置收敛容差
                class_weight=class_weight, # 类别权重
                intercept_scaling=intercept_scaling  # 截距缩放
            )

    # 训练模型
    def fit(self, X, y, sample_weights=None):
        self.model.fit(X, y, sample_weight=sample_weights)
        # 保存模型参数
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_

    # 预测样本属于各个类别的概率
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # 预测样本属于各个类别的对数概率
    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    # 计算决策函数值
    def decision_function(self, X):
        return self.model.decision_function(X)

# 支持向量机分类器类
class SVM():
    # 初始化SVM分类器
    def __init__(self, params):
        # 设置类别权重
        if params['downweight_by_class']:
            class_weight = 'balanced'
        else:
            class_weight = None
        # 创建线性核SVM模型
        self.model = SVC(kernel='linear', class_weight=class_weight)
    
    # 训练模型
    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

    # 计算决策函数值
    def decision_function(self, X):
        return self.model.decision_function(X)

# 工厂函数：根据指定的算法和参数创建二分类器
def build_binary_classifier(algorithm, params):
    # 如果选择逻辑回归
    if algorithm == "logistic_regression":
        assert 'penalty' in params
        penalty = params['penalty']
        # 如果使用L2正则化，返回L2LogisticRegression实例
        if penalty == 'l2':
            return L2LogisticRegression(params)
    # 如果选择SVM，返回SVM实例
    elif algorithm == "svm":
        return SVM(params)
    
# 主函数入口点
if __name__ == "__main__":
    main() 
