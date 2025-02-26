# CellO: *Cell O*ntology-based classification &nbsp; <img src="https://raw.githubusercontent.com/deweylab/CellO/master/cello.png" alt="alt text" width="70px" height="70px">
[# CellO: 基于细胞本体论的分类系统]

![PyPI Version](https://img.shields.io/pypi/v/cello-classify)  

## About
[## 关于]

CellO (Cell Ontology-based classification) is a Python package for performing cell type classification of human RNA-seq data. CellO makes hierarchical predictions against the [Cell Ontology](http://www.obofoundry.org/ontology/cl.html). These classifiers were trained on nearly all of the human primary cell, bulk RNA-seq data in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra).

[CellO (基于细胞本体论的分类)是一个用于对人类RNA-seq数据进行细胞类型分类的Python包。CellO根据[细胞本体论](http://www.obofoundry.org/ontology/cl.html)进行层次化预测。这些分类器是在[序列读取档案库](https://www.ncbi.nlm.nih.gov/sra)中几乎所有的人类原代细胞、整体RNA-seq数据上训练的。]

For more details regarding the underlying method, see the paper:
[Bernstein, M.N., Ma, J., Gleicher, M., Dewey, C.N. (2020). CellO: Comprehensive and hierarchical cell type classification of human cellswith the Cell Ontology. *iScience*, 24(1), 101913.](https://www.sciencedirect.com/science/article/pii/S258900422031110X) 

[有关底层方法的更多详细信息，请参阅论文：
[Bernstein, M.N., Ma, J., Gleicher, M., Dewey, C.N. (2020). CellO：使用细胞本体论对人类细胞进行全面和层次化的细胞类型分类。*iScience*, 24(1), 101913。](https://www.sciencedirect.com/science/article/pii/S258900422031110X)]

There are two modes in which one can use CellO: within Python in conjunction with [Scanpy](), or with the command line. 

[CellO有两种使用模式：在Python中与[Scanpy]()结合使用，或通过命令行使用。]

## Installation
[## 安装]

To install CellO using Pip, run the following command:
[使用Pip安装CellO，运行以下命令：]

`pip install cello-classify`

## Running CellO from within Python
[## 在Python中运行CellO]

CellO's API interfaces with the Scanpy Python library and can integrate into a more general single-cell analysis pipeline. For an example on how to use CellO with Scanpy, please see the [tutorial](https://github.com/deweylab/CellO/blob/package_for_pypi/tutorial/cello_tutorial.ipynb).

[CellO的API与Scanpy Python库对接，可以集成到更通用的单细胞分析流程中。关于如何将CellO与Scanpy一起使用的示例，请参见[教程](https://github.com/deweylab/CellO/blob/package_for_pypi/tutorial/cello_tutorial.ipynb)。]

This tutorial can also be executed from a Google Colab notebook in the cloud: [https://colab.research.google.com/drive/1lNvzrP4bFDkEe1XXKLnO8PZ83StuvyWW?usp=sharing](https://colab.research.google.com/drive/1lNvzrP4bFDkEe1XXKLnO8PZ83StuvyWW?usp=sharing).

这个教程也可以在云端的Google Colab笔记本中执行：[https://colab.research.google.com/drive/1lNvzrP4bFDkEe1XXKLnO8PZ83StuvyWW?usp=sharing](https://colab.research.google.com/drive/1lNvzrP4bFDkEe1XXKLnO8PZ83StuvyWW?usp=sharing)。

## Running CellO from the command line
## 从命令行运行CellO

CellO takes as input a gene expression matrix. CellO accepts data in multiple formats:
* TSV: tab-separated value 
* CSV: comma-separated value
* HDF5: a database in HDF5 format that includes three datasets: a dataset storing the expression matrix, a dataset storing the list of gene-names (i.e. rows), and a gene-set storing the list of cell ID's (i.e. columns)
* 10x formatted directory: a directory in the 10x format including three files: ``matrix.mtx``, ``genes.tsv``, and ``barcodes.tsv``

CellO以基因表达矩阵作为输入。CellO接受多种格式的数据：
* TSV：制表符分隔值
* CSV：逗号分隔值
* HDF5：HDF5格式的数据库，包含三个数据集：存储表达矩阵的数据集、存储基因名称列表（即行）的数据集，以及存储细胞ID列表（即列）的基因集
* 10x格式目录：10x格式的目录，包含三个文件：``matrix.mtx``、``genes.tsv``和``barcodes.tsv``

Given an output-prefix provided to CellO (this can include the path to the output), CellO outputs three tables formatted as tab-separated-value files: 
* ``<output_prefix>.probability.tsv``: a NxM classification probability table of N cells and M cell types where element (i,j) is a probability value that describes CellO's confidence that cell i is of cell type j  
* ``<output_prefix>.binary.tsv``: a NxM binary-decision matrix where element (i,j) is 1 if CellO predicts cell i to be of cell type j and is 0 otherwise.
* ``<output_prefix>.most_specific.tsv``: a table mapping each cell to the most-specific predicted cell
* ``<output_prefix>.log``: a directory that stores log files that store details of CellO's execution
* ``<output_prefix>.log/genes_absent_from_training_set.tsv``: if a new model is trained using the ``-t`` option, then this file will store the genes in CellO's training set that were _not_ found in the input dataset
* ``<output_prefix>.log/clustering.tsv``: a TSV file mapping each cell to its assigned cluster. Note, that if pre-computed clusters are provided via the ``-p`` option, then this file will not be written. 

根据提供给CellO的输出前缀（可以包含输出路径），CellO输出三个制表符分隔值格式的表格文件：
* ``<output_prefix>.probability.tsv``：一个NxM的分类概率表，包含N个细胞和M个细胞类型，其中元素(i,j)是描述CellO对细胞i属于细胞类型j的置信度的概率值
* ``<output_prefix>.binary.tsv``：一个NxM的二元决策矩阵，如果CellO预测细胞i属于细胞类型j，则元素(i,j)为1，否则为0
* ``<output_prefix>.most_specific.tsv``：将每个细胞映射到最具体预测细胞的表格
* ``<output_prefix>.log``：存储CellO执行详细信息的日志文件目录
* ``<output_prefix>.log/genes_absent_from_training_set.tsv``：如果使用``-t``选项训练新模型，则此文件将存储CellO训练集中在输入数据集中未找到的基因
* ``<output_prefix>.log/clustering.tsv``：将每个细胞映射到其分配的簇的TSV文件。注意，如果通过``-p``选项提供预计算的簇，则不会写入此文件

Usage:
[用法：]

```
cello_predict [options] input_file

Options:
  -h, --help            show this help message and exit
  -a ALGO, --algo=ALGO  Hierarchical classification algorithm to apply
                        (default='IR'). Must be one of: 'IR' - Isotonic
                        regression, 'CLR' - cascaded logistic regression
  -d DATA_TYPE, --data_type=DATA_TYPE
                        Data type (required). Must be one of: 'TSV', 'CSV',
                        '10x', or 'HDF5'. Note: if 'HDF5' is used, then
                        arguments must be provided to the h5_cell_key,
                        h5_gene_key, and h5_expression_key parameters.
  -c H5_CELL_KEY, --h5_cell_key=H5_CELL_KEY
                        The key of the dataset within the input HDF5 file
                        specifying which dataset stores the cell ID's.  This
                        argument is only applicable if '-d HDF5' is used
  -g H5_GENE_KEY, --h5_gene_key=H5_GENE_KEY
                        The key of the dataset within the input HDF5 file
                        specifying which dataset stores the gene names/ID's.
                        This argument is only applicable if '-d HDF5' is used
  -e H5_EXPRESSION_KEY, --h5_expression_key=H5_EXPRESSION_KEY
                        The key of the dataset within the input HDF5 file
                        specifying which dataset stores the expression matrix.
                        This argument is only applicable if '-d HDF5' is used
  -r, --rows_cells      Use this flag if expression matrix is organized as
                        CELLS x GENES rather than GENES x CELLS. Not
                        applicable when '-d 10x' is used.
  -u UNITS, --units=UNITS
                        Units of expression. Must be one of: 'COUNTS', 'CPM',
                        'LOG1_CPM', 'TPM', 'LOG1_TPM'
  -s ASSAY, --assay=ASSAY
                        Sequencing assay. Must be one of: '3_PRIME',
                        'FULL_LENGTH'
  -t, --train_model     If the genes in the input matrix don't match what is
                        expected by the classifier, then train a classifier on
                        the input genes. The model will be saved to
                        <output_prefix>.model.dill
  -m MODEL, --model=MODEL
                        Path to pretrained model file.
  -l REMOVE_ANATOMICAL, --remove_anatomical=REMOVE_ANATOMICAL
                        A comma-separated list of terms ID's from the Uberon
                        Ontology specifying which tissues to use to filter
                        results. All cell types known to be resident to the
                        input tissues will be filtered from the results.
  -p PRE_CLUSTERING, --pre_clustering=PRE_CLUSTERING
                        A TSV file with pre-clustered cells. The first column
                        stores the cell names/ID's (i.e. the column names of
                        the input expression matrix) and the second column
                        stores integers referring to each cluster. The TSV
                        file should not have column names.
  -b, --ontology_term_ids
                        Use the less readable, but more rigorous Cell Ontology
                        term id's in output
  -o OUTPUT_PREFIX, --output_prefix=OUTPUT_PREFIX
                        Prefix for all output files. This prefix may contain a
                        path.
```

Notably, the input expression data's genes must match the genes expected by the trained classifier.  If the genes match, then CellO will use a pre-trained classifier to classify the expression profiles (i.e. cells) in the input dataset. 

[需要注意的是，输入表达数据的基因必须与训练好的分类器所期望的基因相匹配。如果基因匹配，那么CellO将使用预训练的分类器对输入数据集中的表达谱（即细胞）进行分类。]

To provide an example, here is how you would run CellO on a toy dataset stored in ``example_input/Zheng_PBMC_10x``. This dataset is a set of 1,000 cells subsampled from the [Zheng et al. (2017)](https://www.nature.com/articles/ncomms14049) dataset.  To run CellO on this dataset, run this command:

[举个例子，这里展示如何在存储在``example_input/Zheng_PBMC_10x``中的示例数据集上运行CellO。这个数据集是从[Zheng等人(2017)](https://www.nature.com/articles/ncomms14049)数据集中抽样的1,000个细胞。要在这个数据集上运行CellO，执行以下命令：]

``cello_predict -d 10x -u COUNTS -s 3_PRIME example_input/Zheng_PBMC_10x -o test``

Note that ``-o test`` specifies the all output files will have the prefix "test". The ``-d`` specifies the input format, ``-u`` specifies the units of the expression matrix, and ``-s`` specifies the assay-type.  For a full list of available formats, units, assay-types, run:

[注意，``-o test``指定所有输出文件都将使用"test"作为前缀。``-d``指定输入格式，``-u``指定表达矩阵的单位，``-s``指定测序类型。要查看可用格式、单位、测序类型的完整列表，运行：]

``cello_predict -h``


### Running CellO with a gene set that is incompatible with a pre-trained model
[### 使用与预训练模型不兼容的基因集运行CellO]

If the genes in the input file do not match the genes on which the model was trained, CellO can be told to train a classifier with only those genes included in the given input dataset by using the ``-t`` flag.  The trained model will be saved to a file named ``<output_prefix>.model.dill`` where ``<output_prefix>`` is the output-prefix argument provided via the ``-o`` option.  Training CellO usually takes under an hour. 

[如果输入文件中的基因与模型训练时使用的基因不匹配，可以通过使用``-t``标志来告诉CellO仅使用给定输入数据集中包含的基因训练分类器。训练好的模型将保存到名为``<output_prefix>.model.dill``的文件中，其中``<output_prefix>``是通过``-o``选项提供的输出前缀。训练CellO通常需要不到一小时。]

For example, to train a model and run CellO on the file ``example_input/LX653_tumor.tsv``, run the command:

[例如，要在文件``example_input/LX653_tumor.tsv``上训练模型并运行CellO，执行以下命令：]

``cello_predict -u COUNTS -s 3_PRIME -t -o test example_input/LX653_tumor.tsv``

Along with the classification results, this command will output a file ``test.model.dill``.

[除了分类结果外，此命令还将输出一个``test.model.dill``文件。]

### Running CellO with a custom model
[### 使用自定义模型运行CellO]

Training a model on a new gene set needs only to be done once (see previous section). For example, to run CellO on ``example_input/LX653_tumor.tsv`` using a specific model stored in a file, run:

[在新的基因集上训练模型只需要进行一次（参见上一节）。例如，要使用存储在文件中的特定模型在``example_input/LX653_tumor.tsv``上运行CellO，执行：]

``cello_predict -u COUNTS -s 3_PRIME -m test.model.dill -o test example_input/LX653_tumor.tsv``

Note that ``-m test.model.dill`` tells CellO to use the model computed in the previous example.

[注意，``-m test.model.dill``告诉CellO使用在前面示例中计算的模型。]

## Quantifying reads with Kallisto to match CellO's pre-trained models
[## 使用Kallisto量化读数以匹配CellO的预训练模型]

We provide a commandline tool for quantifying raw reads with [Kallisto](https://pachterlab.github.io/kallisto/). Note that to run this script, Kallisto must be installed and available in your ``PATH`` environment variable.  This script will output an expression profile that includes all of the genes that CellO is expecting and thus, expression profiles created with this script are automatically compatible with CellO.

[我们提供了一个命令行工具，用于使用[Kallisto](https://pachterlab.github.io/kallisto/)量化原始读数。注意，要运行此脚本，必须安装Kallisto并在你的``PATH``环境变量中可用。此脚本将输出包含CellO期望的所有基因的表达谱，因此，使用此脚本创建的表达谱自动与CellO兼容。]

This script requires a preprocessed kallisto reference.  To download the pre-built Kallisto reference that is compatible with CellO, run the command:

[此脚本需要预处理的kallisto参考。要下载与CellO兼容的预构建Kallisto参考，运行以下命令：]

``bash download_kallisto_reference.sh``

This command will download a directory called ``kallisto_reference`` in the current directory. To run Kallisto on a set of FASTQ files, run the command

[此命令将在当前目录中下载一个名为``kallisto_reference``的目录。要在一组FASTQ文件上运行Kallisto，执行以下命令：]

``cello_quantify_sample <comma_dilimited_fastq_files> <tmp_dir> -o <kallisto_output_file>``

where ``<comma_delimited_fastq_files>`` is a comma-delimited set of FASTQ files containing all of the reads for a single RNA-seq sample and ``<tmp_dir>`` is the location where Kallisto will store it's output files.  The file ``<kallisto_output_file>`` is a tab-separated-value table of the log(TPM+1) values that can be fed directly to CellO.  To run CellO on this output file, run:

[其中``<comma_delimited_fastq_files>``是一组逗号分隔的FASTQ文件，包含单个RNA-seq样本的所有读数，``<tmp_dir>``是Kallisto存储其输出文件的位置。文件``<kallisto_output_file>``是一个制表符分隔值表格，包含可以直接输入到CellO的log(TPM+1)值。要在此输出文件上运行CellO，执行：]

``cell_predict -u LOG1_TPM -s FULL_LENGTH <kallisto_output_file> -o <cell_output_prefix>``

Note that the above command assumes that the assay is a full-length assay (meaning reads can originate from the full-length of the transcript).  If this is a 3-prime assay (reads originate from only the 3'-end of the transcript), the ``-s FULL_LENGTH`` should be replaced with ``-s 3_PRIME`` in the above command.

[注意，上述命令假设测序是全长测序（意味着读数可以来自转录本的全长）。如果这是3'端测序（读数仅来自转录本的3'端），上述命令中的``-s FULL_LENGTH``应替换为``-s 3_PRIME``。]

## Trouble-shooting
[## 故障排除]

If upon running `pip install cello` you receive an error installing Cython, that looks like:

[如果在运行`pip install cello`时收到安装Cython的错误，看起来像这样：]

```
ERROR: Command errored out with exit status 1:
     command: /scratch/cdewey/test_cello/CellO-master/cello_env/bin/python3 -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-wo2dj5q7/quadprog/setup.py'"'"'; __file__='"'"'/tmp/pip-install-wo2dj5q7/quadprog/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base pip-egg-info
         cwd: /tmp/pip-install-wo2dj5q7/quadprog/
    Complete output (5 lines):
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-wo2dj5q7/quadprog/setup.py", line 17, in <module>
        from Cython.Build import cythonize
    ModuleNotFoundError: No module named 'Cython'
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
```

then you may try upgrading to the latest version of pip and Cython by running:

[那么你可以尝试通过运行以下命令升级到最新版本的pip和Cython：]

```
python -m pip install --upgrade pip
pip install --upgrade cython
```
