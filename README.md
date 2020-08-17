# chinese-ner-tf2
中文ner模型使用tensorflow2.1构建

整体结构使用transformer+bi-lstm+crf完成。

训练数据采用clue2020ner数据

## 2020.8.13更新
增加span解码方式

增加TENER框架，参考[TENER](https://github.com/fastnlp/TENER)

[文章思想](https://arxiv.org/abs/1911.04474)

使用TENER目前粗调参数，准确率能到65%上下
