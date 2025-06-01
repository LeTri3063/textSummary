import json
import os

from torch.utils.data import Dataset, DataLoader
import torch


class LCSTSDataset(Dataset):
    """中文文本摘要数据集加载类，继承自PyTorch的Dataset类

    用于加载和处理中文文本摘要数据集，将原始数据转换为模型可处理的格式

    Args:
        file_path (str): 数据集文件路径
        tokenizer: 分词器对象，用于文本编码
        max_length (int, optional): 文本最大长度，默认为512
    """

    def __init__(self, file_path, tokenizer, max_length=512):
        """初始化数据集

        Args:
            file_path: 数据集文件路径
            tokenizer: 分词器对象
            max_length: 输入文本最大长度
        """
        self.tokenizer = tokenizer  # 保存分词器
        self.max_length = max_length  # 保存最大长度设置
        self.data = self._load_data(file_path)  # 加载原始数据

    def _load_data(self, file_path):
        """加载JSON格式的原始数据

        Args:
            file_path: 数据集文件路径

        Returns:
            list: 包含所有数据项的列表
        """
        data = []  # 初始化空列表存储数据
        with open(file_path, 'r', encoding='utf-8') as f:  # 以UTF-8编码打开文件
            temp = json.load(f)  # 加载JSON数据
            for item in temp:  # 遍历每个数据项
                data.append(item)  # 添加到数据列表
        return data  # 返回完整数据列表

    def __len__(self):
        """返回数据集大小

        Returns:
            int: 数据集中的样本数量
        """
        return len(self.data)  # 返回数据列表长度

    def __getitem__(self, idx):
        """获取单个样本数据

        Args:
            idx: 样本索引

        Returns:
            dict: 包含编码后文本、注意力掩码和标签的字典
        """
        item = self.data[idx]  # 获取指定索引的数据项

        # 对原文内容进行编码处理
        text_encoding = self.tokenizer(
            item['content'],  # 原文内容
            max_length=self.max_length,  # 最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 启用截断
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 对摘要内容进行编码处理，最大长度为原文的一半
        summary_encoding = self.tokenizer(
            item['summary'],  # 摘要内容
            max_length=self.max_length // 2,  # 摘要最大长度为原文一半
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 启用截断
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 返回处理后的数据，使用squeeze(0)去除多余的批次维度
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),  # 文本ID
            'attention_mask': text_encoding['attention_mask'].squeeze(0),  # 注意力掩码
            'labels': summary_encoding['input_ids'].squeeze(0)  # 摘要作为标签
        }


def custom_collate_fn(batch):
    """自定义批次处理函数，用于DataLoader

    将多个样本堆叠成一个批次张量

    Args:
        batch: 批次数据列表

    Returns:
        dict: 堆叠后的批次数据字典
    """
    return {
        # 堆叠所有样本的input_ids
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        # 堆叠所有样本的attention_mask
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        # 堆叠所有样本的labels
        'labels': torch.stack([x['labels'] for x in batch])
    }


def create_dataloaders(tokenizer, train_path, val_path, config):
    """创建训练和验证数据加载器

    Args:
        tokenizer: 分词器对象
        train_path: 训练集文件路径
        val_path: 验证集文件路径
        config: 配置对象，包含batch_size等参数

    Returns:
        tuple: (训练数据加载器, 验证数据加载器)
    """
    # 创建训练数据集实例
    train_dataset = LCSTSDataset(train_path, tokenizer, config.max_length)
    # 创建验证数据集实例
    val_dataset = LCSTSDataset(val_path, tokenizer, config.max_length)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=config.batch_size,  # 批次大小
        shuffle=True,  # 打乱数据
        # Windows系统下禁用多进程，其他系统使用4个工作进程
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,  # 启用内存锁页，加速GPU传输
        collate_fn=custom_collate_fn  # 使用自定义的批次处理函数
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,  # 验证数据集
        batch_size=config.batch_size,  # 批次大小
        # Windows系统下禁用多进程，其他系统使用4个工作进程
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,  # 启用内存锁页，加速GPU传输
        collate_fn=custom_collate_fn  # 使用自定义的批次处理函数
    )

    return train_loader, val_loader