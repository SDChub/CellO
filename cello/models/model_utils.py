# 导入pandas用于数据处理
import pandas as pd

# 将标签-分数列表转换为矩阵的工具函数
def convert_to_matrix(label_to_score_list, exps):
    """
    将标签到分数的映射列表转换为矩阵形式
    参数:
        label_to_score_list: 包含多个标签到分数映射字典的列表
        exps: 实验/样本的标识符列表
    返回:
        DataFrame: 行为实验/样本，列为标签的矩阵
    """
    # 收集所有唯一的标签
    all_labels = set()
    for label_to_score in label_to_score_list:
        all_labels.update(label_to_score.keys())
    all_labels = sorted(all_labels)
    
    # 构建分数矩阵
    mat = [
        [
            label_to_score[label]
            for label in all_labels
        ]
        for label_to_score in label_to_score_list
    ]
    
    # 创建DataFrame并返回
    df = pd.DataFrame(
        data=mat,
        index=exps,
        columns=all_labels
    )
    return df

# 主函数入口点
if __name__ == '__main__':
    main()
