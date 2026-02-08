"""
增强版训练脚本 - 适用于本地和Colab长时间训练
自动检测环境并安装依赖
"""

from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
    )
import os
import sys
import json
import math
import time
import torch
import signal
import logging
import subprocess
import importlib
import platform
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ========== 环境检测和依赖安装 ==========


def check_and_install_dependencies():
    """检查并安装必要的依赖包"""

    # 需要安装的包列表
    required_packages = [
        'transformers[torch]>=4.30.0',
        'datasets>=2.14.0',
        'accelerate>=0.21.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        ]

    # 可选包列表（仅用于本地环境）
    optional_packages = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        ]

    print("=" * 60)
    print("检查并安装依赖包")
    print("=" * 60)

    # 检查当前Python版本
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")

    # 检查pip是否可用
    try:
        import pip
        print("✓ pip已安装")
    except ImportError:
        print("❌ pip未安装，请先安装pip")
        sys.exit(1)

    # 检查torch是否已安装（如果未安装，需要根据系统安装合适的版本）
    try:
        import torch
        print(f"✓ PyTorch已安装: {torch.__version__}")
    except ImportError:
        print("⚠ PyTorch未安装，将自动安装...")
        # 根据系统和CUDA版本选择合适的torch版本
        system = platform.system()

        # 检查是否有CUDA
        cuda_available = False
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True)
            cuda_available = result.returncode == 0
        except BaseException:
            pass

        if cuda_available and system == "Linux":
            torch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        elif cuda_available and system == "Windows":
            torch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            torch_package = "torch torchvision torchaudio"

        required_packages = [torch_package] + required_packages

    # 安装所有必需的包
    print("\n安装必需的包:")
    for package in required_packages:
        print(f"正在安装: {package}")
        try:
            # 使用subprocess安装包
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"  ✓ {package.split()[0] if ' ' in package else package}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 安装失败: {package}")
            print(f"    错误: {e}")

    # 可选安装的包（不强制）
    print("\n可选安装的包:")
    for package in optional_packages:
        try:
            # 尝试导入，如果失败则安装
            importlib.import_module(package.split('>=')[0].split('[')[0])
            print(f"  ✓ {package} (已安装)")
        except ImportError:
            print(f"  可选: {package} (未安装，跳过)")

    print("\n✓ 依赖检查完成")
    print("=" * 60)


# ========== Colab环境检测 ==========
IS_COLAB = 'COLAB_GPU' in os.environ

# 如果是Colab环境，安装必要的包
if IS_COLAB:
    print(f"运行环境: Google Colab")

    # 在Colab中安装必要的包
    try:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "transformers[torch]", "datasets", "accelerate", "matplotlib", "tqdm", "--quiet"
            ])
        print("✓ Colab环境依赖安装完成")
    except Exception as e:
        print(f"❌ Colab依赖安装失败: {e}")

    # 自动挂载Google Drive
    from google.colab import drive
    drive_mounted = False
    try:
        drive.mount('/content/drive', force_remount=True)
        drive_mounted = True
        print("✓ Google Drive 挂载成功")
    except Exception as e:
        print(f"Drive挂载失败: {e}")
        print("将使用Colab临时存储")
else:
    print(f"运行环境: 本地环境 ({platform.system()})")

    # 在本地环境中检查并安装依赖
    check_and_install_dependencies()

# ========== 导入Transformers库（在安装后） ==========

# ========== 配置部分 ==========


class TrainingConfig:
    """训练配置类 - 只保留紧急存档，所有文件存储于Google Drive或本地"""

    # 基础路径配置
    if IS_COLAB:
        # Colab环境使用Google Drive
        BASE_DIR = "/content/drive/MyDrive/LLM_Impossible_Training"
    else:
        # 本地环境 - 使用用户主目录下的文件夹
        home_dir = os.path.expanduser("~")
        BASE_DIR = os.path.join(home_dir, "LLM_Impossible_Training")

    # 数据路径
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # 结果保存路径
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # 紧急存档保存路径
    EMERGENCY_DIR = os.path.join(BASE_DIR, "emergency_backups")

    # 日志文件路径
    LOG_FILE = os.path.join(BASE_DIR, "training.log")

    # 训练配置
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-5
    SAVE_STEPS = 1000  # 设置为大值，禁用常规保存
    LOGGING_STEPS = 5
    MAX_CHECKPOINTS = 0  # 不保留常规检查点

    # 数据集配置
    DATASETS = [
        {
            "name": "Natural Language",
            "file_path": os.path.join(DATA_DIR, "data_natural.txt"),
            "model_dir": os.path.join(RESULTS_DIR, "model_natural"),
            "emergency_dir": os.path.join(EMERGENCY_DIR, "natural_language"),
            "storage_location": "drive" if IS_COLAB else "local",
            "resume_checkpoint": None
            },
        {
            "name": "Impossible Language (Reversed)",
            "file_path": os.path.join(DATA_DIR, "impossible_output_reversed.txt"),
            "model_dir": os.path.join(RESULTS_DIR, "model_reversed"),
            "emergency_dir": os.path.join(EMERGENCY_DIR, "reversed"),
            "storage_location": "drive" if IS_COLAB else "local",
            "resume_checkpoint": None
            },
        {
            "name": "Impossible Language (Parity Negation)",
            "file_path": os.path.join(DATA_DIR, "impossible_output_parity_negation.txt"),
            "model_dir": os.path.join(RESULTS_DIR, "model_parity_negation"),
            "emergency_dir": os.path.join(EMERGENCY_DIR, "parity_negation"),
            "storage_location": "drive" if IS_COLAB else "local",
            "resume_checkpoint": None
            }
        ]

    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        # 主目录
        directories = [
            cls.BASE_DIR,
            cls.DATA_DIR,
            cls.RESULTS_DIR,
            cls.EMERGENCY_DIR
            ]

        print("创建目录结构:")
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ {directory}")

        # 为每个数据集创建具体目录
        print("\n创建数据集目录:")
        for dataset in cls.DATASETS:
            os.makedirs(dataset["model_dir"], exist_ok=True)
            os.makedirs(dataset["emergency_dir"], exist_ok=True)
            print(f"  ✓ {dataset['name']}: {dataset['model_dir']}")

# ========== 保持活跃管理器 ==========


class KeepAliveManager:
    @staticmethod
    def simulate_activity():
        """模拟用户活动，防止Colab断开连接"""
        try:
            print(f"[保持活跃] {datetime.now().strftime('%H:%M:%S')} - 训练仍在进行中...")

            # 轻微的内存操作
            dummy_var = [random.random() for _ in range(1000)]
            del dummy_var

            # 检查GPU内存使用情况
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"[GPU内存] 已使用: {gpu_memory:.2f} GB")
            else:
                print("[内存] 使用CPU训练")

            return True
        except Exception as e:
            print(f"[保持活跃] 错误: {e}")
            return False

# ========== 日志配置 ==========


def setup_logging(log_file):
    """设置日志记录"""
    # 确保日志文件所在目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 移除现有的处理器，避免重复
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# ========== 自定义回调函数 - 只保留紧急存档 ==========


class EmergencyOnlyCallback(TrainerCallback):
    """只保留紧急存档的回调函数"""

    def __init__(self, trainer, emergency_dir, model_dir, logger=None):
        self.trainer = trainer
        self.emergency_dir = emergency_dir
        self.model_dir = model_dir
        self.logger = logger
        self.perplexities = []  # 只记录困惑度
        self.start_time = time.time()
        self.last_emergency_time = time.time()
        self.last_keepalive_time = time.time()
        self.emergency_save_interval = 300  # 每5分钟紧急保存一次

        logger.info(f"初始化紧急存档回调，紧急存档目录: {emergency_dir}")
        logger.info("已禁用所有常规检查点，只保留紧急存档")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志时触发 - 只计算和记录困惑度"""
        if logs and 'loss' in logs:
            # 计算困惑度
            loss = logs['loss']
            perplexity = math.exp(loss) if loss < 100 else float('inf')

            # 只添加困惑度到日志
            logs['perplexity'] = perplexity
            self.perplexities.append(perplexity)

            # 日志只显示困惑度
            if self.logger:
                self.logger.info(
                    f"步数 {state.global_step}: 困惑度 = {perplexity:.4f}")

            # 定期模拟活动，防止断开（仅Colab）
            current_time = time.time()
            if IS_COLAB and current_time - self.last_keepalive_time > 300:  # 每5分钟
                KeepAliveManager.simulate_activity()
                self.last_keepalive_time = current_time

            # 定期紧急保存（每5分钟）
            if current_time - self.last_emergency_time > self.emergency_save_interval:
                self._emergency_save(state)
                self.last_emergency_time = current_time

    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步结束时触发"""
        # 每100步紧急保存一次（额外保护）
        if state.global_step % 100 == 0:
            self._emergency_save(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时触发"""
        if self.logger:
            self.logger.info(
                f"Epoch {state.epoch} 完成，总步数: {state.global_step}")
        # 不保存常规检查点，只记录日志

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时触发"""
        if self.logger:
            self.logger.info("训练完成，保存最终模型...")
        self._save_final_model(state)

    def _emergency_save(self, state):
        """紧急存档 - 只保留这一个存档功能"""
        try:
            # 创建紧急存档目录
            timestamp = int(time.time())
            save_dir = os.path.join(
                self.emergency_dir,
                f"emergency_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)

            # 保存状态信息
            state_info = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "save_time": datetime.now().isoformat(),
                "perplexities": self.perplexities[-100:] if self.perplexities else [],
                "total_training_time": time.time() - self.start_time
                }

            with open(os.path.join(save_dir, "emergency_state.json"), 'w') as f:
                json.dump(state_info, f, indent=2)

            # 保存模型（简化版）
            try:
                self.trainer.model.save_pretrained(save_dir)
                if self.trainer.tokenizer is not None:
                    self.trainer.tokenizer.save_pretrained(save_dir)
            except Exception as e:
                self.logger.warning(f"模型保存失败: {e}")

            # 清理旧的紧急存档（只保留最新的3个）
            self._cleanup_old_emergency_backups()

            if self.logger:
                self.logger.info(f"紧急存档保存到: {save_dir}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"紧急存档失败: {e}")

    def _cleanup_old_emergency_backups(self):
        """清理旧的紧急存档，只保留最新的3个"""
        try:
            if not os.path.exists(self.emergency_dir):
                return

            # 获取所有紧急存档目录
            emergency_dirs = []
            for item in os.listdir(self.emergency_dir):
                if item.startswith("emergency_"):
                    item_path = os.path.join(self.emergency_dir, item)
                    if os.path.isdir(item_path):
                        emergency_dirs.append(
                            (item_path, os.path.getmtime(item_path)))

            # 按修改时间排序
            emergency_dirs.sort(key=lambda x: x[1], reverse=True)

            # 删除旧的存档，只保留最新的3个
            for i in range(3, len(emergency_dirs)):
                import shutil
                shutil.rmtree(emergency_dirs[i][0])
                self.logger.info(f"清理旧紧急存档: {emergency_dirs[i][0]}")

        except Exception as e:
            self.logger.error(f"清理紧急存档失败: {e}")

    def _save_final_model(self, state):
        """保存最终模型"""
        try:
            # 最终模型路径
            final_model_path = os.path.join(self.model_dir, "final_model")

            # 清空目录，只保留最终模型
            if os.path.exists(final_model_path):
                import shutil
                shutil.rmtree(final_model_path)

            # 保存最终模型
            self.trainer.model.save_pretrained(final_model_path)
            if self.trainer.tokenizer is not None:
                self.trainer.tokenizer.save_pretrained(final_model_path)

            # 保存困惑度结果
            perplexity_result = {
                "perplexities": self.perplexities,
                "final_perplexity": self.perplexities[-1] if self.perplexities else None,
                "min_perplexity": min(self.perplexities) if self.perplexities else None,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "save_time": datetime.now().isoformat(),
                "total_training_time": time.time() - self.start_time
                }

            with open(os.path.join(final_model_path, "perplexity_results.json"), 'w') as f:
                json.dump(perplexity_result, f, indent=2)

            self.logger.info(f"最终模型保存到: {final_model_path}")

            # 清理紧急存档目录（训练完成）
            if os.path.exists(self.emergency_dir):
                import shutil
                shutil.rmtree(self.emergency_dir)
                self.logger.info(f"训练完成，清理紧急存档目录: {self.emergency_dir}")

        except Exception as e:
            self.logger.error(f"保存最终模型失败: {e}")

# ========== 训练函数 - 只保留紧急存档 ==========


def train_with_emergency_only(
        config, dataset_config, model_name, tokenizer, logger):
    """只保留紧急存档的训练函数"""

    logger.info(f"开始训练: {model_name}")
    logger.info(f"数据文件: {dataset_config['file_path']}")
    logger.info(f"最终模型将保存到: {dataset_config['model_dir']}")
    logger.info(f"紧急存档将保存到: {dataset_config['emergency_dir']}")
    logger.info("注意: 已禁用常规检查点，只保留紧急存档")

    # 检查数据文件是否存在
    if not os.path.exists(dataset_config['file_path']):
        logger.error(f"数据文件不存在: {dataset_config['file_path']}")
        logger.info("请确保数据文件已放置在正确位置")
        logger.info(f"数据文件应该位于: {dataset_config['file_path']}")
        return None, None

    # 创建模型输出目录
    model_dir = dataset_config['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    # 检查是否已有最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    if os.path.exists(final_model_path):
        logger.info(f"发现已有最终模型: {final_model_path}")
        if IS_COLAB:
            # 在Colab中询问用户
            choice = input(f"模型 '{model_name}' 已训练完成，是否重新训练？(y/N): ")
            if choice.lower() != 'y':
                logger.info(f"跳过模型: {model_name}")

                # 加载已有的困惑度结果
                try:
                    result_file = os.path.join(
                        final_model_path, "perplexity_results.json")
                    with open(result_file, 'r') as f:
                        existing_results = json.load(f)
                    return [], existing_results.get("perplexities", [])
                except BaseException:
                    return [], []
        else:
            # 在本地环境中，默认不重新训练
            logger.info(f"跳过已训练的模型: {model_name}")
            try:
                result_file = os.path.join(
                    final_model_path, "perplexity_results.json")
                with open(result_file, 'r') as f:
                    existing_results = json.load(f)
                return [], existing_results.get("perplexities", [])
            except BaseException:
                return [], []

    # 检查是否有可恢复的紧急存档
    resume_from_checkpoint = None
    emergency_dir = dataset_config['emergency_dir']

    if os.path.exists(emergency_dir):
        # 查找最新的紧急存档
        emergency_dirs = []
        for item in os.listdir(emergency_dir):
            if item.startswith("emergency_"):
                item_path = os.path.join(emergency_dir, item)
                if os.path.isdir(item_path):
                    emergency_dirs.append(
                        (item_path, os.path.getmtime(item_path)))

        if emergency_dirs:
            # 按修改时间排序，获取最新的紧急存档
            emergency_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_emergency = emergency_dirs[0][0]
            resume_from_checkpoint = latest_emergency
            logger.info(f"发现紧急存档: {resume_from_checkpoint}")

            # 加载训练状态
            state_file = os.path.join(latest_emergency, "emergency_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_info = json.load(f)
                logger.info(
                    f"恢复训练状态: 步数={state_info.get('global_step', 0)}, epoch={state_info.get('epoch', 0)}")

    # 加载模型
    try:
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            model = GPT2LMHeadModel.from_pretrained(resume_from_checkpoint)
            logger.info(f"从紧急存档恢复训练: {resume_from_checkpoint}")
        else:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            logger.info("创建新的GPT-2模型")
    except Exception as e:
        logger.error(f"加载模型失败: {e}，创建新模型")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    # 准备数据集
    try:
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=dataset_config['file_path'],
            block_size=128
            )
        logger.info(f"数据集加载成功，样本数: {len(train_dataset)}")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return None, None

    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        )

    # 训练参数 - 禁用所有常规检查点
    training_args = TrainingArguments(
        output_dir=dataset_config['emergency_dir'],  # 输出目录设置为紧急存档目录
        overwrite_output_dir=False,  # 不覆盖，保留紧急存档
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,  # 设置为大值，禁用常规保存
        save_total_limit=config.MAX_CHECKPOINTS,  # 设置为0，不保留常规检查点
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        max_steps=700,
        warmup_steps=100,
        save_strategy="no",  # 禁用保存策略
        load_best_model_at_end=False,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        disable_tqdm=False,
        logging_first_step=True,
        logging_dir=os.path.join(model_dir, "logs"),
        eval_strategy="no",
        save_safetensors=False,
        save_on_each_node=False,
        no_cuda=not torch.cuda.is_available(),
        )

    # 创建紧急存档目录
    emergency_dir = dataset_config['emergency_dir']
    os.makedirs(emergency_dir, exist_ok=True)

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        )

    # 添加只保留紧急存档的回调
    emergency_callback = EmergencyOnlyCallback(
        trainer, emergency_dir, model_dir, logger=logger)
    trainer.add_callback(emergency_callback)

    # 开始训练
    try:
        logger.info("开始训练过程...")
        logger.info("=" * 60)
        logger.info("训练配置：")
        logger.info(f"1. 最终模型将保存到: {model_dir}")
        logger.info(f"2. 紧急存档将保存到: {emergency_dir}")
        logger.info(f"3. 已禁用所有常规检查点，只保留紧急存档")
        logger.info(f"4. 紧急存档每5分钟自动保存一次")
        logger.info(f"5. 如果训练中断，可以从最新紧急存档恢复")
        logger.info("=" * 60)

        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint)

        # 收集训练指标
        perplexities = emergency_callback.perplexities

        logger.info(f"训练完成: {model_name}")
        if perplexities:
            logger.info(f"最终困惑度: {perplexities[-1]:.4f}")
            logger.info(f"最小困惑度: {min(perplexities):.4f}")
            logger.info(
                f"困惑度下降: {(perplexities[0] - perplexities[-1]):.4f}" if len(perplexities) > 1 else "N/A")

        return perplexities, train_result

    except KeyboardInterrupt:
        logger.warning(f"训练被用户中断: {model_name}")
        # 紧急保存
        emergency_callback._emergency_save(trainer.state)
        return None, None

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # 尝试紧急保存
        try:
            emergency_callback._emergency_save(trainer.state)
        except BaseException:
            pass

        return None, None

# ========== 主函数 ==========


def main():
    """主训练函数 - 只保留紧急存档"""

    print("=" * 80)
    print("训练脚本 - 不可能语言实验")
    print(f"运行环境: {'Google Colab' if IS_COLAB else '本地环境'}")
    print("只保留紧急存档")
    print("=" * 80)

    # 环境信息
    if not IS_COLAB:
        print("\n📊 本地环境信息:")
        print(f"   操作系统: {platform.system()} {platform.release()}")
        print(f"   Python版本: {platform.python_version()}")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 重要提示
    print("\n⚠️  存储说明：")
    if IS_COLAB:
        print("   • 所有模型都只保存在Google Drive")
    else:
        print(f"   • 所有模型保存在: {TrainingConfig.BASE_DIR}")
    print("   • 已禁用所有常规检查点，只保留紧急存档")
    print("   • 紧急存档每5分钟自动保存一次")
    print("   • 训练完成后，只保留最终模型")
    print("=" * 80)

    # 创建目录结构
    config = TrainingConfig
    config.create_directories()

    # 初始化日志记录器
    logger = setup_logging(config.LOG_FILE)
    logger.info("训练脚本开始执行 - 只保留紧急存档")
    logger.info(f"工作目录: {config.BASE_DIR}")
    logger.info("已禁用常规检查点，只保留紧急存档")

    # 初始化分词器
    logger.info("初始化分词器...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ 分词器初始化成功")
    except Exception as e:
        logger.error(f"分词器初始化失败: {e}")
        print(f"❌ 分词器初始化失败，请检查网络连接")
        return

    # 训练所有数据集
    all_metrics = {}

    for dataset_config in config.DATASETS:
        dataset_name = dataset_config['name']

        print(f"\n{'=' * 60}")
        print(f"训练数据集: {dataset_name}")
        print(f"存储位置: {'Google Drive' if IS_COLAB else '本地硬盘'}")
        print(f"存档策略: 只保留紧急存档")
        print(f"{'=' * 60}")

        # 开始训练计时
        start_time = time.time()

        # 训练模型
        perplexities, train_result = train_with_emergency_only(
            config, dataset_config, dataset_name, tokenizer, logger
            )

        # 计算训练时间
        training_time = time.time() - start_time

        if perplexities is not None:
            # 保存训练结果
            metrics = {
                "perplexities": perplexities,
                "final_perplexity": perplexities[-1] if perplexities else None,
                "min_perplexity": min(perplexities) if perplexities else None,
                "training_time": training_time,
                "total_steps": len(perplexities),
                "dataset": dataset_name,
                "file_path": dataset_config['file_path'],
                "model_dir": dataset_config['model_dir'],
                "completed_at": datetime.now().isoformat()
                }

            all_metrics[dataset_name] = metrics

            # 保存结果到文件
            result_file = os.path.join(
                dataset_config['model_dir'],
                "perplexity_results.json")
            with open(result_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"困惑度结果保存到: {result_file}")

            # 打印训练统计
            print(f"\n✅ 训练完成 - {dataset_name}:")
            print(
                f"   最终困惑度: {perplexities[-1]:.4f}" if perplexities else "   最终困惑度: N/A")
            print(
                f"   最小困惑度: {min(perplexities):.4f}" if perplexities else "   最小困惑度: N/A")
            print(f"   训练步数: {len(perplexities)}")
            print(f"   训练时间: {training_time / 60:.2f} 分钟")

        else:
            logger.error(f"❌ 训练失败: {dataset_name}")
            all_metrics[dataset_name] = None

    # 打印最终汇总
    print(f"\n{'=' * 80}")
    print("训练完成汇总")
    print(f"{'=' * 80}")

    for dataset_name, metrics in all_metrics.items():
        if metrics:
            final_ppl = metrics['final_perplexity']
            min_ppl = metrics['min_perplexity']
            steps = metrics['total_steps']
            time_min = metrics['training_time'] / 60

            print(f"✅ {dataset_name}:")
            print(f"   最终困惑度: {final_ppl:.4f}")
            print(f"   最小困惑度: {min_ppl:.4f}")
            print(f"   训练步数: {steps}")
            print(f"   训练时间: {time_min:.1f}分钟")
            print(f"   存储位置: {metrics['model_dir']}")
        else:
            print(f"❌ {dataset_name}: 训练失败")

    print(f"{'=' * 80}")

    # 环境特定提示
    if IS_COLAB:
        print("\n📊 训练状态汇总:")
        print(f"   日志文件: {config.LOG_FILE}")
        print(f"   所有模型保存在: {config.RESULTS_DIR}")

        print("\n📁 模型目录结构:")
        print(f"   {config.RESULTS_DIR}/")
        for dataset in config.DATASETS:
            print(f"   ├── {os.path.basename(dataset['model_dir'])}")
            print(f"   │   └── final_model/ (最终模型)")
    else:
        print("\n📊 训练状态汇总:")
        print(f"   日志文件: {config.LOG_FILE}")
        print(f"   所有模型保存在: {config.RESULTS_DIR}")
        print(f"   数据目录: {config.DATA_DIR}")

        print("\n📁 本地目录结构:")
        print(f"   {config.BASE_DIR}/")
        print(f"   ├── data/ (数据文件)")
        print(f"   ├── results/ (训练结果)")
        print(f"   │   ├── model_natural/")
        print(f"   │   ├── model_reversed/")
        print(f"   │   └── model_parity_negation/")
        print(f"   ├── emergency_backups/ (紧急存档)")
        print(f"   └── training.log (日志文件)")

    print("\n⚠️  存档策略说明:")
    print("   已禁用所有常规检查点")
    print("   只保留紧急存档（每5分钟自动保存）")
    print("   训练完成后，只保留最终模型，清理紧急存档")

    logger.info(f"训练脚本执行完成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🏁 训练脚本执行完成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ========== 脚本入口 ==========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

        # 保存错误信息
        try:
            error_file = os.path.join(TrainingConfig.BASE_DIR, "error_log.txt")
            with open(error_file, 'a') as f:  # 追加模式
                f.write(f"\n{'=' * 60}\n")
                f.write(f"错误时间: {datetime.now().isoformat()}\n")
                f.write(f"错误信息: {str(e)}\n\n")
                f.write(traceback.format_exc())

            print(f"📝 错误日志已追加到: {error_file}")
        except BaseException:
            pass
