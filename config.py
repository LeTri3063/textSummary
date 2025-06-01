from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型结构配置类"""
    vocab_size: int  # 必须提供的词汇表大小参数
    batch_size: int = 16  # 批处理大小
    d_model: int = 512  # 模型维度
    nhead: int = 8  # 注意力头数
    num_encoder_layers: int = 6  # 编码器层数
    num_decoder_layers: int = 6  # 解码器层数
    dropout: float = 0.1  # dropout率
    max_length: int = 512  # 最大序列长度
    activation: str = "gelu"  # 激活函数


@dataclass
class TrainingConfig:
    """训练过程配置类"""
    batch_size: int = 16  # 训练批大小
    epochs: int = 10  # 训练轮数
    learning_rate: float = 1e-4  # 初始学习率
    warmup_steps: int = 4000  # 学习率预热步数
    accumulation_steps: int = 4  # 梯度累积步数
    max_grad_norm: float = 1.0  # 梯度裁剪最大值
    weight_decay: float = 0.01  # L2正则化系数
    adam_epsilon: float = 1e-8  # Adam优化器的epsilon


@dataclass
class GenerationConfig:
    """文本生成配置类"""
    max_length: int = 150  # 生成的最大长度
    min_length: int = 10  # 生成的最小长度
    top_k: int = 50  # top-k采样参数
    top_p: float = 0.95  # 核采样参数
    temperature: float = 0.7  # 温度参数
    repetition_penalty: float = 1.2  # 重复惩罚系数
    num_beams: int = 1  # beam search的beam数
    do_sample: bool = True  # 是否使用采样


@dataclass
class TestConfig:
    """测试配置参数"""
    test_data_path: str = "lcsts_data/test.json"  # 测试集路径
    model_path: str = "best_model.pt"  # 模型保存路径
    results_path: str = "test_results.json"  # 结果保存路径
    batch_size: int = 16  # 测试批次大小
    max_length: int = 512  # 最大输入长度
    num_workers: int = 1  # 数据加载工作进程数
    rouge_metrics: tuple = ("rouge1", "rouge2", "rougeL")  # ROUGE评估指标
    use_stemmer: bool = True  # 是否使用词干分析