import h5py
expr_matrix_f = 'resources/training_set/log_tpm_10x_genes.h5'
print('Loading expression data from {}...'.format(expr_matrix_f))
with h5py.File(expr_matrix_f, 'r') as f:
    # 处理实验ID
    the_exps = [
        str(x)[2:-1]
        for x in f['experiment'][:]
    ]
    # 处理基因ID
    gene_ids = [
        str(x)[2:-1]
        for x in f['gene_id'][:]
    ]
    # 加载表达数据矩阵
    data_matrix = f['expression'][:]
print('Loaded matrix of shape {}'.format(data_matrix.shape))
print('done.')