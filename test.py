import os
import json
import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import BertTokenizer
from config import ModelConfig, TestConfig, GenerationConfig
from model import DynamicTransformerSummarizer
from data_loader import LCSTSDataset


class ModelTester:
    """模型测试类"""

    def __init__(self, test_config: TestConfig):
        """初始化测试器"""
        self.config = test_config
        self.device = self._setup_device()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", legacy=False)
        self.model = self._load_model()
        self.scorer = rouge_scorer.RougeScorer(
            self.config.rouge_metrics,
            use_stemmer=self.config.use_stemmer
        )

    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device

    def _load_model(self) -> DynamicTransformerSummarizer:
        """加载训练好的模型"""
        model_config = ModelConfig(vocab_size=self.tokenizer.vocab_size)
        model = DynamicTransformerSummarizer(model_config).to(self.device)
        model.tokenizer = self.tokenizer

        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _create_dataloader(self) -> torch.utils.data.DataLoader:
        """创建测试数据加载器"""
        dataset = LCSTSDataset(
            file_path=self.config.test_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0 if os.name == 'nt' else self.config.num_workers,
            collate_fn=lambda x: x
        )

    def run_test(self, gen_config: GenerationConfig = None) -> dict:
        """执行完整测试流程"""
        test_loader = self._create_dataloader()
        results = self._evaluate(test_loader, gen_config)
        self._save_results(results)
        return results

    def _evaluate(self, dataloader, gen_config: GenerationConfig = None) -> dict:
        """评估模型性能"""
        gen_config = gen_config or GenerationConfig()
        scores = {metric: [] for metric in self.config.rouge_metrics}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                # 生成预测
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gen_config=gen_config
                )

                # 解码文本
                preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # 计算指标
                for pred, ref in zip(preds, refs):
                    result = self.scorer.score(pred, ref)
                    for metric in scores.keys():
                        scores[metric].append(result[metric].fmeasure)

        return {f"rouge-{k.split('rouge')[-1].lower()}": np.mean(v) for k, v in scores.items()}

    def _save_results(self, results: dict):
        """保存测试结果"""
        with open(self.config.results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {self.config.results_path}")


def main():
    """主测试函数"""
    # 初始化配置
    test_config = TestConfig()

    # 创建并运行测试器
    tester = ModelTester(test_config)
    results = tester.run_test()

    # 打印结果
    print("\nTest Results:")
    for metric, score in results.items():
        print(f"{metric.upper()}: {score:.4f}")


if __name__ == "__main__":
    main()