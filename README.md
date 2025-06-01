# 基于BERT和Transformer的文本摘要系统

系统主要由以下几个部分组成：
```
main.py - 程序入口，负责初始化模型、数据加载器和训练器
trainer.py - 训练和评估逻辑
model.py - 模型定义
data_loader.py - 数据加载和预处理
config.py - 各种配置参数
```

## 一些说明
- 本实验是在‌LCSTS中文短文本摘要数据集上进行；
- 所使用的计算资源是AutoDL的远程服务器NVIDIA GeForce RTX 4090(24GB)；
- 代码中的注释主要由deepseek生成；
