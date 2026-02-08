from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import json
import os
import logging
import math
import torch
from tqdm import tqdm
import accelerate

os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# add padding symbol
tokenizer.pad_token = tokenizer.eos_token

class PerplexityCallback:
    def __init__(self, trainer, eval_steps=10):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.perplexities = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_step = state.global_step
            if current_step % self.eval_steps == 0:
                loss = logs['loss']
                perplexity = math.exp(loss) if loss < 100 else float('inf')  #prevent memory leak
                self.perplexities.append(perplexity)
                self.steps.append(current_step)
                logs['perplexity'] = perplexity

def calculate_test_perplexity(model, tokenizer, file_path, max_length=128, batch_size=4):
    if not os.path.exists(file_path):
        return float('inf')
    
    # 读取测试数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        return float('inf')
    
    # 创建数据集
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=max_length
    )
    
    # 创建DataLoader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        # 分批次计算损失
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_collated = data_collator(batch)
            
            input_ids = batch_collated["input_ids"].to(model.device)
            attention_mask = batch_collated.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            labels = input_ids.clone()
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if loss is not None:
                # 计算实际token数量（排除padding）
                if attention_mask is not None:
                    token_count = attention_mask.sum().item()
                else:
                    token_count = input_ids.numel()
                
                total_loss += loss.item() * token_count
                total_tokens += token_count
    
    # 计算平均损失和Perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
    else:
        return float('inf')

def train_and_record(file_path, output_dir, dataset_name):
    """训练并记录各项指标"""
    # 创建测试文件路径（假设测试文件与训练文件在同一目录，但带有_test后缀）
    test_file_path = file_path.replace('.txt', '_test.txt')
    if not os.path.exists(test_file_path):
        # 如果测试文件不存在，使用训练文件的一部分作为测试集
        test_file_path = file_path
    
    # 训练模型
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,       # train for five rounds
        per_device_train_batch_size=4,
        logging_steps=10,         # record loss in every ten rounds
        learning_rate=5e-5,
        save_strategy="epoch",    
        save_total_limit=2,       
        report_to="none",         
        evaluation_strategy="no",  # 我们不使用内置的评估
    )
    
    # 创建自定义回调
    perplexity_callback = PerplexityCallback(None, eval_steps=10)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 将回调函数附加到trainer
    perplexity_callback.trainer = trainer
    trainer.add_callback(perplexity_callback)
    
    print(f"Now training {dataset_name}...")
    train_output = trainer.train()
    
    # 保存模型
    trainer.save_model()
    print(f"{dataset_name} model has been saved to: {output_dir}")
    
    # 处理历史日志
    log_history = trainer.state.log_history
    losses = [x['loss'] for x in log_history if 'loss' in x]
    
    # 从回调中获取perplexities
    perplexities = perplexity_callback.perplexities
    
    # 计算最终测试集Perplexity
    print(f"Calculating final test perplexity for {dataset_name}...")
    final_test_perplexity = calculate_test_perplexity(model, tokenizer, test_file_path)
    
    loss_file = os.path.join(output_dir, "training_metrics.json")
    with open(loss_file, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "file_path": file_path,
            "test_file_path": test_file_path,
            "losses": losses,
            "perplexities": perplexities,
            "final_loss": losses[-1] if losses else None,
            "final_test_perplexity": final_test_perplexity,
            "total_steps": len(losses)
        }, f, indent=2)
    
    return losses, perplexities, final_test_perplexity

def plot_all_metrics_comparison(metrics_dict):
    """绘制所有数据集的损失和困惑度对比图"""
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 绘制损失曲线
    for i, (dataset_name, metrics) in enumerate(metrics_dict.items()):
        if 'losses' in metrics and metrics['losses']:
            ax1.plot(metrics['losses'], label=dataset_name, 
                    color=colors[i % len(colors)], alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Training Steps (x10)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison: All Datasets', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 绘制Perplexity曲线
    for i, (dataset_name, metrics) in enumerate(metrics_dict.items()):
        if 'perplexities' in metrics and metrics['perplexities']:
            ax2.plot(metrics['perplexities'], label=dataset_name, 
                    color=colors[i % len(colors)], alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Training Steps (x10)', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Training Perplexity Comparison: All Datasets', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/training_metrics_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 单独绘制最终测试Perplexity的柱状图
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    dataset_names = []
    final_perplexities = []
    
    for dataset_name, metrics in metrics_dict.items():
        if 'final_test_perplexity' in metrics and metrics['final_test_perplexity'] < float('inf'):
            dataset_names.append(dataset_name)
            final_perplexities.append(metrics['final_test_perplexity'])
    
    if dataset_names:
        bars = ax3.bar(dataset_names, final_perplexities, color=colors[:len(dataset_names)])
        ax3.set_xlabel('Dataset', fontsize=12)
        ax3.set_ylabel('Final Test Perplexity', fontsize=12)
        ax3.set_title('Final Test Perplexity Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, final_perplexities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('./results/final_test_perplexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_natural_vs_impossible_comparisons(natural_metrics, impossible_metrics_dict):
    """分别绘制自然语言与每个不可能语言变体的对比图"""
    natural_name = "Natural Language"
    
    for imp_name, imp_metrics in impossible_metrics_dict.items():
        # 创建两个子图：损失和困惑度
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绘制损失对比
        if 'losses' in natural_metrics and natural_metrics['losses']:
            ax1.plot(natural_metrics['losses'], label=natural_name, 
                    color='blue', alpha=0.8, linewidth=2)
        
        if 'losses' in imp_metrics and imp_metrics['losses']:
            ax1.plot(imp_metrics['losses'], label=imp_name, 
                    color='red', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Training Steps (x10)', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Loss: {natural_name} vs {imp_name}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 绘制困惑度对比
        if 'perplexities' in natural_metrics and natural_metrics['perplexities']:
            ax2.plot(natural_metrics['perplexities'], label=natural_name, 
                    color='blue', alpha=0.8, linewidth=2)
        
        if 'perplexities' in imp_metrics and imp_metrics['perplexities']:
            ax2.plot(imp_metrics['perplexities'], label=imp_name, 
                    color='red', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Training Steps (x10)', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title(f'Perplexity: {natural_name} vs {imp_name}', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = imp_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f'./results/comparison_natural_vs_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    os.makedirs("./results", exist_ok=True)
    
    datasets_config = [
        {
            "name": "Natural Language",
            "file_path": "./data/data_natural.txt",
            "output_dir": "./results/model_natural"
        },
        {
            "name": "Impossible Language (Reversed)",
            "file_path": "./data/impossible_output_reversed.txt",
            "output_dir": "./results/model_reversed"
        },
        {
            "name": "Impossible Language (Parity Negation)",
            "file_path": "./data/impossible_output_parity_negation.txt",
            "output_dir": "./results/model_parity_negation"
        }
    ]
    
    all_metrics = {}
    
    for config in datasets_config:
        os.makedirs(config["output_dir"], exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"Data file: {config['file_path']}")
        print(f"Model will be saved to: {config['output_dir']}")
        print(f"{'='*60}")
        
        losses, perplexities, final_test_perplexity = train_and_record(
            config["file_path"], 
            config["output_dir"],
            config["name"]
        )
        
        all_metrics[config["name"]] = {
            "losses": losses,
            "perplexities": perplexities,
            "final_test_perplexity": final_test_perplexity
        }
    
    print(f"\n{'='*60}")
    print("All training completed!")
    print(f"{'='*60}")
    
    print("\nTraining Statistics:")
    print(f"{'Dataset':<40} {'Steps':<10} {'Final Loss':<12} {'Final Test Perplexity':<20}")
    print("-" * 85)
    
    for dataset_name, metrics in all_metrics.items():
        final_loss = metrics['losses'][-1] if metrics['losses'] else "N/A"
        final_perplexity = metrics['final_test_perplexity']
        steps = len(metrics['losses'])
        
        if final_perplexity < float('inf'):
            print(f"{dataset_name:<40} {steps:<10} {final_loss:<12.4f} {final_perplexity:<20.4f}")
        else:
            print(f"{dataset_name:<40} {steps:<10} {final_loss:<12.4f} {'N/A':<20}")
    
    # 绘制所有指标的对比图
    plot_all_metrics_comparison(all_metrics)
    
    natural_metrics = all_metrics["Natural Language"]
    impossible_metrics_dict = {
        name: metrics for name, metrics in all_metrics.items() 
        if name != "Natural Language"
    }
    
    plot_natural_vs_impossible_comparisons(natural_metrics, impossible_metrics_dict)
    
    # 保存汇总统计
    summary_file = "./results/training_summary.json"
    summary_data = {}
    
    for dataset_name, metrics in all_metrics.items():
        summary_data[dataset_name] = {
            "final_loss": metrics['losses'][-1] if metrics['losses'] else None,
            "final_test_perplexity": metrics['final_test_perplexity'],
            "training_steps": len(metrics['losses']),
            "min_loss": min(metrics['losses']) if metrics['losses'] else None,
            "max_perplexity": max(metrics['perplexities']) if metrics['perplexities'] else None,
            "min_perplexity": min(metrics['perplexities']) if metrics['perplexities'] else None
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary statistics saved to: {summary_file}")
    print("\n" + "="*60)
    print("All plots have been saved to ./results/ directory")
    print("="*60)

if __name__ == "__main__":
    main()
