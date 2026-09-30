# GloVe 与 IMDb 情感分析实践

本项目记录一次导师布置的模型运行实践：将已有的 GloVe 词向量转换为 Gensim 可读取的格式，再运行 PyTorch 编写的 IMDb 情感分类代码，体验数据预处理、模型训练和预测的完整流程。

## 项目背景与个人工作

导师提供了 GloVe 模型和一整套 `imdb_sentiment_analysis_torch` 代码，并要求从 GitHub 寻找 `glove-gensim.py`，进行少量转换后使用模型，再运行情感分析代码。

在这次实践中，我使用 DeepSeek 大模型辅助修改了从 GitHub 下载的 `glove-gensim.py`，以适配 Python 3.14 环境；同时借助大模型修改了 `imdb_process.py` 和 `imdb_cnn.py`，完成运行并保存了预测结果。本仓库保留这三个脚本及已有结果，作为学习过程的记录。

原始模型、转换脚本和情感分析代码并非本人从零编写。原始转换脚本的具体 GitHub 地址、导师提供代码的上游地址和许可证尚未记录，待确认后补充；现有代码中的来源注释予以保留。本仓库未为这些第三方代码另行指定许可证。

## 仓库内容

```text
GloVe/
├── README.md
├── .gitignore
├── requirements.txt
├── resource/
│   ├── README.md
│   └── glove-gensim.py
├── model/
│   └── README.md
└── imdb_sentiment_analysis_torch/
    ├── imdb_process.py
    ├── imdb_cnn.py
    └── result/
        └── cnn.csv
```

`resource` 中的模型文本、压缩包及原始代码压缩包不上传；`model/glove_model.txt` 不上传。两处目录中的说明文件记录资源用途及准备方式。

情感分析目录仅保留本次修改的两个 Python 文件和运行结果；其余模型示例、语料、预处理缓存和 Python 字节码不上传。本地已有文件不受忽略规则影响。

## 环境与验证范围

本次适配的目标环境为 Python 3.14。`requirements.txt` 根据脚本导入整理，未锁定版本，也不代表一套已验证的 Python 3.14 依赖组合。安装时需要选择与本机 Python、操作系统及 CPU 或 CUDA 环境匹配的依赖版本。

整理上传时，三个脚本通过了 Python 3.14 语法解析检查，结果 CSV 也已检查。此次整理未重新安装依赖、加载完整词向量或重跑训练，因此不把语法检查等同于完整运行验证，也不能据此确定已有预测文件生成时的具体环境。

在已准备好的 Python 环境中，从仓库根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 资源准备与运行

### 准备 GloVe 词向量

将 `glove.840B.300d.txt` 放入 `resource/`。可以使用导师提供的文件，也可从 [Stanford GloVe 官方页面](https://nlp.stanford.edu/projects/glove/) 获取对应的 840B、300 维词向量。详见 [资源说明](resource/README.md)。

转换脚本使用当前工作目录中的文件，需进入 `resource` 运行：

```powershell
Set-Location resource
python glove-gensim.py
Move-Item -LiteralPath glove_model.txt -Destination ../model/glove_model.txt
Set-Location ..
```

如果 `model/glove_model.txt` 已经存在，可直接使用已有文件并跳过转换。脚本先生成转换文件，再用 Gensim 加载整个模型进行相似度演示；模型较大，需要充足的磁盘空间和内存。详见 [模型说明](model/README.md)。

### 准备 IMDb 数据

将以下三个文件放入 `imdb_sentiment_analysis_torch/corpus/imdb/`，目录不存在时先创建：

- `labeledTrainData.tsv`：带标签的影评，用于划分训练集和验证集。
- `testData.tsv`：待预测的影评。
- `unlabeledTrainData.tsv`：现有预处理脚本会读取该文件，虽然后续流程未使用，运行时仍需提供。

可使用导师提供的数据；这些文件名对应 [Kaggle 的 Bag of Words Meets Bags of Popcorn 数据页面](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)。下载可能需要登录并接受该数据集的使用条件。

### 预处理与训练

从仓库根目录依次运行：

```powershell
python imdb_sentiment_analysis_torch/imdb_process.py
python imdb_sentiment_analysis_torch/imdb_cnn.py
```

第一步清洗影评、构建词表、将序列截断或填充到 512 个词，并生成 `imdb_sentiment_analysis_torch/pickle/imdb_glove.pickle3`。第二步使用固定的 300 维词嵌入和一维 CNN 进行分类，默认训练 10 轮、批大小 256，自动选择可用的 CUDA 或 CPU，最后写入 `result/cnn.csv`。重新运行会覆盖同名结果，需保留时请先备份。

## 已有运行结果

仓库中的 [cnn.csv](imdb_sentiment_analysis_torch/result/cnn.csv) 是本次实践已保存的预测文件，包含 `id` 和 `sentiment` 两列，共 25,000 条记录。其中标签 1 有 11,246 条，标签 0 有 13,754 条；按 IMDb 二分类约定分别表示正面和负面。

该 CSV 是预测输出，不是准确率报告。仓库未包含完整训练日志或测试集真实标签，因此不报告测试准确率。当前代码未统一固定所有随机种子，重新运行的预测也可能不同。
