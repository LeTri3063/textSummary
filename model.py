import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import ModelConfig, GenerationConfig
from contextlib import contextmanager


@contextmanager
def no_init_weights():
    """上下文管理器，用于临时禁用模型权重初始化

    用于在模型初始化时不自动初始化权重，常用于模型并行或延迟初始化场景
    """
    old_init = torch.nn.Module.__init__  # 保存原始的Module初始化方法
    # 临时替换为空的初始化方法
    torch.nn.Module.__init__ = lambda *args, **kwargs: None
    try:
        yield  # 在此上下文内执行的代码将使用空的初始化方法
    finally:
        torch.nn.Module.__init__ = old_init  # 恢复原始的初始化方法


class DynamicTransformerSummarizer(nn.Module):
    """基于BERT和Transformer的动态文本摘要模型"""

    def __init__(self, config: ModelConfig):
        """初始化模型

        Args:
            config (ModelConfig): 模型配置对象
        """
        super().__init__()

        self.config = config  # 保存配置
        self.tokenizer = None  # 分词器将在外部设置

        # 安全初始化BERT模型
        self.bert = self._init_bert()

        # 初始化Transformer模型
        self.transformer = nn.Transformer(
            d_model=config.d_model,  # 模型维度
            nhead=config.nhead,  # 注意力头数
            num_encoder_layers=config.num_encoder_layers,  # 编码器层数
            num_decoder_layers=config.num_decoder_layers,  # 解码器层数
            dropout=config.dropout,  # Dropout率
            batch_first=True,  # 输入格式为(batch, seq, feature)
            device='cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
        )

        # 投影层：将BERT的768维输出投影到模型指定维度
        self.dynamic_embedding = nn.Linear(768, config.d_model)
        # 输出层：将模型输出投影到词汇表大小
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

        # 初始化自定义层的权重
        self._init_weights()

    def _init_bert(self):
        """初始化预训练的BERT模型

        Returns:
            BertModel: 预训练的中文BERT模型
        """
        return BertModel.from_pretrained("bert-base-chinese")

    def _init_weights(self):
        """使用Xavier均匀分布初始化自定义层的权重"""
        for module in [self.dynamic_embedding, self.output_layer]:
            nn.init.xavier_uniform_(module.weight)  # 初始化权重
            if module.bias is not None:
                module.bias.data.zero_()  # 偏置初始化为0

    def forward(self, input_ids, attention_mask, labels=None):
        """模型前向传播

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 训练时的目标标签(可选)

        Returns:
            训练模式返回logits，生成模式返回生成的token序列
        """
        # 使用BERT编码输入文本
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state  # 获取最后一层隐藏状态

        # 将BERT输出投影到模型维度
        encoder_output = self.dynamic_embedding(bert_output)

        if labels is not None:
            # 训练模式 - 使用教师强制(teacher forcing)
            tgt = self._shift_right(labels)  # 右移目标序列
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))  # 生成解码器掩码

            # 通过Transformer模型
            output = self.transformer(
                src=encoder_output,  # 编码器输出
                tgt=self.dynamic_embedding(self.bert.embeddings.word_embeddings(tgt)),  # 解码器输入
                src_key_padding_mask=~attention_mask.bool(),  # 编码器padding掩码
                tgt_mask=tgt_mask.to(input_ids.device),  # 解码器自回归掩码
                memory_key_padding_mask=~attention_mask.bool()  # 编码器输出padding掩码
            )
            return self.output_layer(output)  # 返回预测logits
        else:
            # 生成模式 - 自回归生成文本
            return self.generate(encoder_output, attention_mask)

    def generate(self, encoder_output, attention_mask, gen_config=None):
        """自回归文本生成

        Args:
            encoder_output: 编码器输出
            attention_mask: 编码器注意力掩码
            gen_config: 生成配置(可选)

        Returns:
            生成的token序列
        """
        gen_config = gen_config or GenerationConfig()  # 使用默认配置如果没有提供
        batch_size = encoder_output.size(0)  # 获取批次大小
        device = encoder_output.device  # 获取设备信息

        # 初始化解码器输入为[CLS] token
        decoder_input = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,  # 使用CLS作为起始token
            dtype=torch.long,
            device=device
        )

        # 自回归生成循环
        for _ in range(gen_config.max_length):
            # 生成三角掩码防止解码器看到未来信息
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1))

            # 解码器前向传播
            output = self.transformer.decoder(
                tgt=self.dynamic_embedding(self.bert.embeddings.word_embeddings(decoder_input)),
                memory=encoder_output,  # 编码器输出作为memory
                tgt_mask=tgt_mask.to(device),  # 自回归掩码
                memory_key_padding_mask=~attention_mask.bool()  # 编码器padding掩码
            )

            # 获取最后一个token的logits
            logits = self.output_layer(output[:, -1])
            # 采样下一个token
            next_token = self._sample_next_token(logits, gen_config, decoder_input)
            # 将新token添加到序列中
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # 如果所有序列都生成了[SEP] token则提前终止
            if (decoder_input == self.tokenizer.sep_token_id).any(dim=1).all():
                break

        return decoder_input

    def _sample_next_token(self, logits, gen_config, prev_tokens):
        """根据生成配置采样下一个token

        Args:
            logits: 模型输出的原始logits
            gen_config: 生成配置
            prev_tokens: 之前生成的tokens

        Returns:
            下一个token的ID
        """
        # 应用温度调节
        logits = logits / gen_config.temperature

        # 应用重复惩罚
        if gen_config.repetition_penalty != 1.0:
            score = torch.gather(logits, 1, prev_tokens)  # 获取已生成token的分数
            # 根据惩罚系数调整分数
            score = torch.where(score < 0, score * gen_config.repetition_penalty,
                                score / gen_config.repetition_penalty)
            logits.scatter_(1, prev_tokens, score)  # 更新logits

        # Top-k采样
        if gen_config.top_k > 0:
            # 保留top_k个最高概率的token
            indices_to_remove = logits < torch.topk(logits, gen_config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')  # 屏蔽其他token

        # Top-p(核)采样
        if gen_config.top_p > 0:
            # 按概率排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 计算累积概率
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 找到第一个超过p的累积概率
            sorted_indices_to_remove = cumulative_probs > gen_config.top_p
            # 保留第一个超过p的token之前的所有token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # 应用屏蔽
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # 从剩余token中采样
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # 多项式采样

    def _shift_right(self, input_ids):
        """将目标序列右移，用于教师强制训练

        Args:
            input_ids: 原始目标序列

        Returns:
            右移后的序列(开头添加[CLS] token)
        """
        shifted_input = input_ids.new_zeros(input_ids.shape)  # 创建相同形状的零张量
        shifted_input[:, 1:] = input_ids[:, :-1].clone()  # 右移序列
        shifted_input[:, 0] = self.tokenizer.cls_token_id  # 开头设置为[CLS]
        return shifted_input

    def _generate_square_subsequent_mask(self, sz):
        """生成自回归掩码(上三角矩阵)

        Args:
            sz: 序列长度

        Returns:
            sz x sz的上三角矩阵，对角线以上为负无穷
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)