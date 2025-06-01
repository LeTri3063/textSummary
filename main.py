import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import BertTokenizer
from config import ModelConfig, TrainingConfig
from data_loader import create_dataloaders
from model import DynamicTransformerSummarizer
from trainer import Trainer


def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def main():
    # 环境检查 - Windows系统下禁用DataLoader多进程
    if os.name == 'nt':
        print("Windows系统检测到，已自动禁用DataLoader多进程")

    device = setup_device()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", legacy=False)

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)  # 使用分词器的词汇表大小
    training_config = TrainingConfig()  # 使用默认训练配置

    train_loader, val_loader = create_dataloaders(
        tokenizer,  # 分词器
        train_path='lcsts_data/train.json',  # 训练集路径
        val_path='lcsts_data/valid.json',  # 验证集路径
        config=model_config  # 模型配置
    )

    model = DynamicTransformerSummarizer(model_config).to(device)
    # 设置模型的分词器(用于生成时的文本处理)
    model.tokenizer = tokenizer

    trainer = Trainer(
        model=model,  # 待训练模型
        train_loader=train_loader,  # 训练数据加载器
        val_loader=val_loader,  # 验证数据加载器
        config=training_config,  # 训练配置
        device=device  # 计算设备
    )
    trainer.train()


if __name__ == "__main__":
    main()