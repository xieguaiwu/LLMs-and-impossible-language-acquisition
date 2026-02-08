#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 自然语言处理模型 - 语言建模训练
参考 colab_train_babydataset.py 的结构
在 baby_data/ 目录的数据集上训练，记录loss和perplexity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import glob


# ========== 配置部分 ==========


class TrainingConfig:
    """训练配置类"""

    # 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "baby_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_baby")
    LOG_FILE = os.path.join(BASE_DIR, "training_baby.log")

    # 训练配置
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MAX_STEPS = 2000
    LOGGING_STEPS = 5
    SAVE_STEPS = 100

    # 模型配置
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    MAX_LENGTH = 128

    # 数据集配置
    DATASETS = [
        {
            "name": "Natural Language",
            "data_dir": os.path.join(DATA_DIR, "natural"),
            "model_dir": os.path.join(RESULTS_DIR, "model_natural"),
            "file_pattern": "*.train"
        },
        {
            "name": "Impossible Language (Parity Negation)",
            "data_dir": os.path.join(DATA_DIR, "parity_negation"),
            "model_dir": os.path.join(RESULTS_DIR, "model_parity_negation"),
            "file_pattern": "*.train"
        },
        {
            "name": "Impossible Language (Reversed)",
            "data_dir": os.path.join(DATA_DIR, "reversed"),
            "model_dir": os.path.join(RESULTS_DIR, "model_reversed"),
            "file_pattern": "*.train"
        }
    ]

    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        directories = [cls.BASE_DIR, cls.DATA_DIR, cls.RESULTS_DIR]
        print("创建目录结构:")
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ {directory}")

        for dataset in cls.DATASETS:
            os.makedirs(dataset["model_dir"], exist_ok=True)
            print(f"  ✓ {dataset['name']}: {dataset['model_dir']}")


# ========== 日志配置 ==========


def setup_logging(log_file):
    """设置日志记录"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


import logging


# ========== 数据集类 ==========


class LanguageModelingDataset(Dataset):
    """语言建模数据集类 - 惰性加载版本"""

    def __init__(self, data_dir, file_pattern, vocab, max_length=128, max_samples=None):
        self.vocab = vocab
        self.max_length = max_length
        self.max_samples = max_samples

        # 查找所有匹配的文件
        pattern = os.path.join(data_dir, file_pattern)
        self.data_files = glob.glob(pattern)

        if not self.data_files:
            raise FileNotFoundError(f"在目录 {data_dir} 中没有找到匹配 {file_pattern} 的文件")

        # 统计总行数（不加载内容）
        self.file_line_counts = []
        self.total_lines = 0
        for file_path in self.data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            self.file_line_counts.append(line_count)
            self.total_lines += line_count

        # 使用总行数作为样本数上限
        self.num_samples = min(self.total_lines, max_samples) if max_samples else self.total_lines

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机选择一个文件和行
        file_idx = np.random.randint(0, len(self.data_files))
        file_path = self.data_files[file_idx]
        line_count = self.file_line_counts[file_idx]

        # 随机选择一行
        line_idx = np.random.randint(0, line_count)

        # 读取指定行
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    break

        line = line.strip()
        if not line:
            # 如果行为空，返回一个填充的样本
            input_seq = [self.vocab['<PAD>']] * self.max_length
            target = self.vocab['<PAD>']
            return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

        # 分词（按字符）
        tokens = list(line)

        # 转换为索引
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        # 随机选择一个位置生成输入-目标对
        if len(indices) > 1:
            pos = np.random.randint(0, len(indices) - 1)
            input_seq = indices[:pos+1]
            target = indices[pos+1]
        else:
            input_seq = indices
            target = self.vocab['<UNK>']

        # 填充到固定长度
        if len(input_seq) < self.max_length:
            input_seq = input_seq + [self.vocab['<PAD>']] * (self.max_length - len(input_seq))
        else:
            input_seq = input_seq[-self.max_length:]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ========== LSTM 语言模型 ==========


class LSTMLanguageModel(nn.Module):
    """LSTM 语言模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(LSTMLanguageModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]

        # LSTM 前向传播
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时间步的输出
        last_output = lstm_output[:, -1, :]  # [batch_size, hidden_dim]

        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # [batch_size, vocab_size]

        return logits


# ========== 训练类 ==========


class LSTMLanguageModelTrainer:
    """LSTM 语言模型训练器"""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.losses = []
        self.perplexities = []
        self.start_time = time.time()

    def build_vocab(self, data_dir, file_pattern, min_freq=2, max_lines=100000):
        """构建词汇表"""
        from collections import Counter

        # 查找所有匹配的文件
        pattern = os.path.join(data_dir, file_pattern)
        data_files = glob.glob(pattern)

        # 统计词频（限制最大行数以节省内存）
        word_freq = Counter()
        lines_processed = 0

        for file_path in data_files:
            if lines_processed >= max_lines:
                break
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if lines_processed >= max_lines:
                        break
                    line = line.strip()
                    if line:
                        tokens = list(line)
                        word_freq.update(tokens)
                    lines_processed += 1

        if self.logger:
            self.logger.info(f"构建词汇表：处理了 {lines_processed} 行，发现 {len(word_freq)} 个唯一字符")

        # 构建词汇表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        vocab_size = len(vocab)
        if self.logger:
            self.logger.info(f"词汇表大小: {vocab_size}")

        return vocab

    def train(self, dataset_config):
        """训练模型"""

        dataset_name = dataset_config['name']
        data_dir = dataset_config['data_dir']
        model_dir = dataset_config['model_dir']
        file_pattern = dataset_config.get('file_pattern', '*.train')

        if self.logger:
            self.logger.info(f"开始训练: {dataset_name}")
            self.logger.info(f"数据目录: {data_dir}")

        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            if self.logger:
                self.logger.error(f"数据目录不存在: {data_dir}")
            return None, None

        # 构建词汇表
        vocab = self.build_vocab(data_dir, file_pattern)
        vocab_size = len(vocab)

        # 创建数据集（限制最大样本数以节省内存）
        max_samples = min(50000, self.config.MAX_STEPS * self.config.BATCH_SIZE)
        dataset = LanguageModelingDataset(
            data_dir,
            file_pattern,
            vocab,
            max_length=self.config.MAX_LENGTH,
            max_samples=max_samples
        )

        if self.logger:
            self.logger.info(f"数据集大小: {len(dataset)} 样本")

        # 划分训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # 初始化模型
        model = LSTMLanguageModel(
            vocab_size,
            self.config.EMBEDDING_DIM,
            self.config.HIDDEN_DIM,
            self.config.NUM_LAYERS,
            self.config.DROPOUT
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

        # 训练循环
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练阶段
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_inputs, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}'):
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # 记录loss和perplexity
                if global_step % self.config.LOGGING_STEPS == 0:
                    avg_loss = epoch_loss / epoch_steps
                    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

                    self.losses.append(avg_loss)
                    self.perplexities.append(perplexity)

                    if self.logger:
                        self.logger.info(
                            f"步数 {global_step}: loss = {avg_loss:.4f}, 困惑度 = {perplexity:.4f}"
                        )

                # 定期保存检查点
                if global_step % self.config.SAVE_STEPS == 0:
                    self._save_checkpoint(
                        model, optimizer, vocab, global_step, epoch, model_dir
                    )

                # 检查是否达到最大步数
                if global_step >= self.config.MAX_STEPS:
                    break

            if global_step >= self.config.MAX_STEPS:
                break

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_steps += 1

            val_loss /= val_steps
            val_perplexity = math.exp(val_loss) if val_loss < 100 else float('inf')

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}"
                )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_best_model(model, vocab, model_dir, val_loss, val_perplexity)

        # 保存最终模型
        self._save_final_model(model, vocab, model_dir)

        # 保存训练结果
        training_result = {
            "losses": self.losses,
            "perplexities": self.perplexities,
            "final_loss": self.losses[-1] if self.losses else None,
            "final_perplexity": self.perplexities[-1] if self.perplexities else None,
            "min_loss": min(self.losses) if self.losses else None,
            "min_perplexity": min(self.perplexities) if self.perplexities else None,
            "total_steps": global_step,
            "total_training_time": time.time() - self.start_time,
            "completed_at": datetime.now().isoformat()
        }

        result_file = os.path.join(model_dir, "training_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(training_result, f, indent=2)

        if self.logger:
            self.logger.info(f"训练结果保存到: {result_file}")

        return {"losses": self.losses, "perplexities": self.perplexities}, training_result

    def _save_checkpoint(self, model, optimizer, vocab, step, epoch, model_dir):
        """保存检查点（不保存完整历史以节省内存）"""
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab
        }, checkpoint_path)

    def _save_best_model(self, model, vocab, model_dir, val_loss, val_perplexity):
        """保存最佳模型"""
        best_model_path = os.path.join(model_dir, "best_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity
        }, best_model_path)

    def _save_final_model(self, model, vocab, model_dir):
        """保存最终模型"""
        final_model_path = os.path.join(model_dir, "final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab
        }, final_model_path)


# ========== 主函数 ==========


def main():
    """主训练函数"""

    print("=" * 80)
    print("LSTM 语言模型训练 - baby_data 数据集")
    print("=" * 80)

    # 环境信息
    print("\n📊 环境信息:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 配置
    config = TrainingConfig
    config.create_directories()

    # 初始化日志记录器
    logger = setup_logging(config.LOG_FILE)
    logger.info("LSTM 语言模型训练开始")
    logger.info(f"工作目录: {config.BASE_DIR}")

    # 创建训练器
    trainer = LSTMLanguageModelTrainer(config, logger=logger)

    # 训练所有数据集
    all_metrics = {}

    for dataset_config in config.DATASETS:
        dataset_name = dataset_config['name']

        print(f"\n{'=' * 60}")
        print(f"训练数据集: {dataset_name}")
        print(f"数据目录: {dataset_config['data_dir']}")
        print(f"{'=' * 60}")

        # 开始训练计时
        start_time = time.time()

        # 训练模型
        training_data, train_result = trainer.train(dataset_config)

        # 计算训练时间
        training_time = time.time() - start_time

        if training_data is not None:
            losses = training_data.get("losses", [])
            perplexities = training_data.get("perplexities", [])

            # 打印训练统计
            print(f"\n✅ 训练完成 - {dataset_name}:")
            if losses and perplexities:
                print(f"   最终loss: {losses[-1]:.4f}, 最终困惑度: {perplexities[-1]:.4f}")
                print(f"   最小loss: {min(losses):.4f}, 最小困惑度: {min(perplexities):.4f}")
                print(f"   训练步数: {len(losses)}")
                print(f"   训练时间: {training_time / 60:.2f} 分钟")
            else:
                print("   没有训练数据")
        else:
            print(f"❌ 训练失败: {dataset_name}")

    # 打印最终汇总
    print(f"\n{'=' * 80}")
    print("训练完成汇总")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
