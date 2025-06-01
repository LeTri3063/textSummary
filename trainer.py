import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
from rouge_score import rouge_scorer
import numpy as np
import json
import matplotlib.pyplot as plt
import os


class Trainer:
    """模型训练和评估类"""

    def __init__(self, model, train_loader, val_loader, config, device):
        """初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置对象
            device: 训练设备(cpu/cuda)
        """
        self.model = model  # 待训练模型
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.config = config  # 训练配置
        self.device = device  # 训练设备
        self.optimizer = self._create_optimizer()  # 创建优化器
        self.scheduler = self._create_scheduler()  # 创建学习率调度器
        # 初始化ROUGE评估器(计算ROUGE-1和ROUGE-L)
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.best_score = -1  # 记录最佳ROUGE分数
        self.train_loss_history = []  # 记录训练损失历史
        self.val_loss_history = []  # 记录验证损失历史
        self.rouge1_history = []  # 记录ROUGE-1分数历史
        self.rougeL_history = []  # 记录ROUGE-L分数历史

    def _create_optimizer(self):
        """创建AdamW优化器，对部分参数禁用权重衰减"""
        # 不需要权重衰减的参数(偏置和LayerNorm参数)
        no_decay = ['bias', 'LayerNorm.weight']
        # 将参数分为两组：一组应用权重衰减，另一组不应用
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,  # 应用权重衰减
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,  # 禁用权重衰减
            }
        ]
        # 创建AdamW优化器
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,  # 学习率
            eps=self.config.adam_epsilon  # Adam epsilon参数
        )

    def _create_scheduler(self):
        """创建学习率调度器，包含warmup阶段"""

        def lr_lambda(current_step):
            """计算学习率比例因子"""
            # Warmup阶段：线性增加学习率
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            # Warmup后阶段：线性衰减学习率
            return max(
                0.0,
                float(self.config.epochs * len(self.train_loader) - current_step) / float(
                    max(1, self.config.epochs * len(self.train_loader) - self.config.warmup_steps
                        )
                )
            )

            # 创建LambdaLR调度器

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """执行模型训练"""
        for epoch in range(self.config.epochs):  # 遍历每个epoch
            self.model.train()  # 设置为训练模式
            epoch_loss = 0  # 记录epoch累计损失
            # 使用进度条显示训练过程
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')

            for step, batch in enumerate(progress_bar):
                # 将数据移动到指定设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                # 计算损失(忽略padding token)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),  # 展平预测结果
                    batch['labels'].view(-1),  # 展平标签
                    ignore_index=self.model.tokenizer.pad_token_id  # 忽略padding token
                )

                # 梯度累积：按累积步数缩放损失
                loss = loss / self.config.accumulation_steps
                loss.backward()  # 反向传播
                epoch_loss += loss.item()  # 累计损失

                # 达到累积步数时更新参数
                if (step + 1) % self.config.accumulation_steps == 0:
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm  # 最大梯度范数
                    )
                    self.optimizer.step()  # 参数更新
                    self.scheduler.step()  # 学习率更新
                    self.optimizer.zero_grad()  # 梯度清零

                # 更新进度条显示信息
                progress_bar.set_postfix({
                    'loss': epoch_loss / (step + 1),  # 平均损失
                    'lr': self.scheduler.get_last_lr()[0]  # 当前学习率
                })

            # 计算并记录平均训练损失
            avg_train_loss = epoch_loss / len(self.train_loader)
            self.train_loss_history.append(avg_train_loss)

            # 每个epoch结束后进行验证
            val_results = self.evaluate()
            print(f"Validation Results - Epoch {epoch + 1}:")
            print(f"ROUGE-1: {val_results['rouge-1']:.4f}")  # ROUGE-1分数
            print(f"ROUGE-L: {val_results['rouge-l']:.4f}")  # ROUGE-L分数

            # 记录验证损失和ROUGE分数
            self.val_loss_history.append(val_results['val_loss'])
            self.rouge1_history.append(val_results['rouge-1'])
            self.rougeL_history.append(val_results['rouge-l'])

            # 保存最佳模型(基于ROUGE-L分数)
            if val_results['rouge-l'] > self.best_score:
                self.best_score = val_results['rouge-l']
                self.save_model('best_model.pt')  # 保存模型

        # 训练结束后保存损失和指标数据
        self.save_training_metrics()
        # 绘制训练曲线
        self.plot_training_curves()

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()  # 设置为评估模式
        # 初始化ROUGE评估器
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge1_scores = []  # 存储ROUGE-1分数
        rougeL_scores = []  # 存储ROUGE-L分数
        val_loss = 0.0  # 验证损失

        with torch.no_grad():  # 禁用梯度计算
            # 使用进度条显示评估过程
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                # 将数据移动到指定设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 计算验证损失
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch['labels'].view(-1),
                    ignore_index=self.model.tokenizer.pad_token_id
                )
                val_loss += loss.item()

                # 生成预测文本
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.config.max_target_length  # 最大生成长度
                )

                # 将token ids转换为文本
                batch_preds = self.model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True  # 跳过特殊token
                )
                batch_labels = self.model.tokenizer.batch_decode(
                    batch['labels'],
                    skip_special_tokens=True
                )

                # 计算每个预测的ROUGE分数
                for pred, label in zip(batch_preds, batch_labels):
                    scores = scorer.score(pred, label)  # 计算ROUGE分数
                    rouge1_scores.append(scores['rouge1'].fmeasure)  # ROUGE-1 F1分数
                    rougeL_scores.append(scores['rougeL'].fmeasure)  # ROUGE-L F1分数

        # 计算平均验证损失
        avg_val_loss = val_loss / len(self.val_loader)

        # 返回评估结果
        return {
            'val_loss': avg_val_loss,  # 平均验证损失
            'rouge-1': np.mean(rouge1_scores),  # 平均ROUGE-1分数
            'rouge-l': np.mean(rougeL_scores)  # 平均ROUGE-L分数
        }

    def save_model(self, path):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),  # 模型参数
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
            'scheduler_state_dict': self.scheduler.state_dict(),  # 调度器状态
            'best_score': self.best_score,  # 最佳ROUGE分数
            'config': self.config  # 训练配置
        }, path)

    def save_training_metrics(self):
        """保存训练指标到JSON文件"""
        metrics = {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'rouge1': self.rouge1_history,
            'rougeL': self.rougeL_history
        }

        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        metrics_path = os.path.join(self.config.output_dir, 'training_metrics.json')

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Training metrics saved to {metrics_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 8))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # 绘制ROUGE分数曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.rouge1_history, label='ROUGE-1')
        plt.plot(self.rougeL_history, label='ROUGE-L')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('ROUGE Scores')
        plt.legend()

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.config.output_dir, 'training_curves.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Training curves saved to {plot_path}")