import spacy
import os
from pathlib import Path
import re

# 增加Spacy的最大文本长度限制
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # 增加到200万字符


class ImpossibleGrammarFactory:
    def __init__(self):
        pass

    def rule_reverse(self, sentence):
        # 移除标点
        text = sentence.rstrip('.?!')
        words = text.split()
        return " ".join(words[::-1]) + "."

        def rule_parity_negation(self, sentence):
        text = sentence.rstrip('.?!')
        words = text.split()

        if len(words) % 2 == 0:
            return "Not " + " ".join(words) + "."
        else:
            return " ".join(words) + " Not."


def split_into_chunks(text, chunk_size=500000):
    """将长文本分割成较小的块，尽量在句子边界处分割"""
    chunks = []
    current_chunk = ""
    
    # 按句子分割（简单的句子分割）
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def process_single_sentence(sentence, factory):
    if not sentence.strip():
        return "", "", ""
    
    original = sentence.strip()
    
    # 如果句子过长，截断处理
    if len(original) > 1000:
        original = original[:500] + original[-500:] if len(original) > 1000 else original
    
    try:
        reversed_sentence = factory.rule_reverse(original)
        parity_sentence = factory.rule_parity_negation(original)
    except Exception as e:
        # 如果处理出错，返回简化版本
        words = original.rstrip('.?!').split()[:10]  # 只取前10个词
        simplified = " ".join(words) + "."
        reversed_sentence = factory.rule_reverse(simplified)
        parity_sentence = factory.rule_parity_negation(simplified)
    
    return reversed_sentence, parity_sentence


def process_large_text(text, factory, chunk_size=500000):
    """处理大段文本，分割成较小的块后分别处理"""
    # 将文本分割成较小的块
    chunks = split_into_chunks(text, chunk_size)
    
    reversed_results = []
    parity_results = []
    
    total_chunks = len(chunks)
    print(f"  将文本分割成 {total_chunks} 个块...")
    
    for i, chunk in enumerate(chunks):
        print(f"  处理第 {i+1}/{total_chunks} 个块...")
        
        # 使用spacy分句处理每个块
        try:
            doc = nlp(chunk)
            for sent in doc.sents:
                rev, fix, par = process_single_sentence(sent.text, factory)
                if rev:  # 只添加非空结果
                    reversed_results.append(rev)
                    parity_results.append(par)
        except Exception as e:
            print(f"    处理块时出错: {e}")
            # 如果spacy处理失败，使用简单句子分割
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            for sent in sentences:
                if sent.strip():
                    rev, fix, par = process_single_sentence(sent, factory)
                    if rev:
                        reversed_results.append(rev)
                        parity_results.append(par)
    
    return reversed_results,  parity_results


def process_file(input_file_path, output_dir, factory):
    """处理单个文件"""
    print(f"处理文件: {input_file_path}")
    
    try:
        # 读取文件并统计大小
        file_size = os.path.getsize(input_file_path)
        print(f"  文件大小: {file_size / (1024*1024):.2f} MB")
        
        # 分批读取大文件
        batch_size = 50 * 1024 * 1024  # 50MB一批
        all_reversed = []
        all_parity = []
        
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            batch = ""
            batch_num = 1
            
            while True:
                chunk = f.read(batch_size)
                if not chunk:
                    break
                
                batch += chunk
                print(f"  处理批次 {batch_num}...")
                
                # 处理当前批次
                rev, fix, par = process_large_text(batch, factory)
                all_reversed.extend(rev)
                all_parity.extend(par)
                
                # 重置批次（保留最后一部分以避免切断句子）
                if len(batch) > 10000:
                    # 找到最后一个句子结束的位置
                    last_sentence_end = max(
                        batch.rfind('. '),
                        batch.rfind('? '),
                        batch.rfind('! '),
                        batch.rfind('\n')
                    )
                    if last_sentence_end > 0:
                        batch = batch[last_sentence_end+1:]
                    else:
                        batch = batch[-5000:]  # 保留最后5000字符
                
                batch_num += 1
        
        # 准备输出文件名
        file_stem = Path(input_file_path).stem
        reversed_file = os.path.join(output_dir, f"{file_stem}_reversed.train")
        parity_file = os.path.join(output_dir, f"{file_stem}_parity_negation.train")
        
        # 写入反转规则结果
        with open(reversed_file, 'w', encoding='utf-8') as f:
            for sentence in all_reversed:
                f.write(sentence + "\n")
        
        # 写入奇偶否定结果
        with open(parity_file, 'w', encoding='utf-8') as f:
            for sentence in all_parity:
                f.write(sentence + "\n")
        
        print(f"  已生成:")
        print(f"    - {reversed_file}: {len(all_reversed)} 个句子")
        print(f"    - {parity_file}: {len(all_parity)} 个句子")
        
        return len(all_reversed)
        
    except Exception as e:
        print(f"  处理文件失败: {e}")
        return 0


def process_directory(input_dir, output_dir):
    """处理整个目录"""
    factory = ImpossibleGrammarFactory()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.train文件
    train_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.train'):
            train_files.append(os.path.join(input_dir, file))
    
    print(f"在目录 {input_dir} 中找到 {len(train_files)} 个.train文件")
    print("=" * 60)
    
    total_sentences = 0
    
    # 处理每个文件
    for input_file in train_files:
        sentence_count = process_file(input_file, output_dir, factory)
        if sentence_count:
            total_sentences += sentence_count
        print("-" * 40)
    
    print("=" * 60)
    print(f"处理完成!")
    print(f"总计处理了 {len(train_files)} 个文件")
    print(f"总计生成了 {total_sentences} 个转换后的句子")
    
    # 显示输出文件统计
    print("\n输出文件统计:")
    print("-" * 40)
    for file in os.listdir(output_dir):
        if file.endswith('.train'):
            file_path = os.path.join(output_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"{file}: {line_count} 行")
            except:
                print(f"{file}: 读取失败")


def demo_sample_transformations():
    """演示转换规则"""
    factory = ImpossibleGrammarFactory()
    
    sample_sentences = [
        "The black cat is eating a fresh fish.",
        "She can read books quickly.",
        "They have been waiting for hours.",
        "The students will study mathematics."
    ]
    
    print("转换规则演示:")
    print("=" * 60)
    
    for i, sentence in enumerate(sample_sentences, 1):
        print(f"\n原始句子 {i}: {sentence}")
        print(f"  反转规则: {factory.rule_reverse(sentence)}")
        print(f"  奇偶否定: {factory.rule_parity_negation(sentence)}")


if __name__ == "__main__":
    print("=" * 60)
    print("大规模不可能语法生成器")
    print("=" * 60)
    
    # 演示转换规则
    demo_sample_transformations()
    
    print("\n" + "=" * 60)
    print("开始处理数据集...")
    print("=" * 60)
    
    # 配置输入输出目录
    input_directory = "train_100M"  # 你的数据集目录
    output_directory = "impossible_transformed"  # 输出目录
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录 '{input_directory}' 不存在")
        print("请确保数据集目录存在并包含以下文件:")
        print("  - bnc_spoken.train")
        print("  - childes.train")
        print("  - gutenberg.train")
        print("  - open_subtitles.train")
        print("  - simple_wiki.train")
        print("  - switchboard.train")
    else:
        # 处理整个目录
        process_directory(input_directory, output_directory)
