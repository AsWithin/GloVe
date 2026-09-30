# 资源文件说明

此目录仅上传 `glove-gensim.py` 和本说明。以下本地资源不放入 GitHub：

| 文件 | 用途 | 本地文件大小 |
| --- | --- | --- |
| `glove.840B.300d.zip` | 原始 GloVe 词向量压缩包 | 2,176,768,976 字节 |
| `glove.840B.300d.txt` | 解压后的 300 维词向量，转换脚本的输入 | 5,646,239,124 字节 |
| `imdb_sentiment_analysis_torch.zip` | 导师提供的完整情感分析代码压缩包 | 28,483 字节 |

GloVe 资源可从 [Stanford GloVe 官方页面](https://nlp.stanford.edu/projects/glove/) 获取，选择 Common Crawl 840B、300d 版本。导师提供的原始代码压缩包没有已确认的公开下载地址；仓库只保存此次修改并使用的两个情感分析脚本。

将词向量文本放在本目录后，在本目录运行 `python glove-gensim.py`。脚本为文本添加词表大小与维度的首行，生成当前目录下的 `glove_model.txt`，再通过 Gensim 加载并演示词相似度。后续 IMDb 预处理需要将输出放入上一级 `model/` 目录。

本地大文件仍可留在这里使用，`.gitignore` 会将它们排除。转换脚本最初从 GitHub 下载，经 DeepSeek 辅助适配；具体上游链接待补充。
