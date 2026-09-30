# 转换后词向量说明

此目录用于存放 `glove_model.txt`。该文件在本地为 5,648,435,155 字节，未上传至 GitHub，已加入忽略规则。

它是 GloVe 预训练词向量的 Word2Vec 文本格式版本，供 Gensim 的 `KeyedVectors.load_word2vec_format` 读取，并非 IMDb CNN 训练得到的权重文件。

使用 `resource/glove-gensim.py` 转换 `glove.840B.300d.txt` 后，将生成的 `glove_model.txt` 放到本目录。脚本写入的首行为：

```text
2196017 300
```

两个数字分别表示脚本预设的词向量条目数和每条向量的维度。正文保留原始词向量。

`imdb_sentiment_analysis_torch/imdb_process.py` 会通过脚本相对路径查找本目录中的文件。若已有可用的转换结果，可直接放入此处，不必再次转换。完整准备及运行步骤见[项目说明](../README.md)。
