import spacy

nlp = spacy.load("en_core_web_sm")


class ImpossibleGrammarFactory:
    def __init__(self):
        pass

    def rule_reverse(self, sentence):
        #remove puncuation
        text = sentence.rstrip('.?!')
        words = text.split()
        return " ".join(words[::-1]) + "."

    def rule_fixed_distance_insert(self, sentence, target_pos="VERB", offset=2, marker="glorp"):
        doc = nlp(sentence)

        target_token = None
        for token in doc:
            if token.pos_ == target_pos:
                target_token = token
                break

        if target_token is None:
            text = sentence.rstrip('.?!')
            return text + " " + marker + "."

        words = []
        target_in_words_index = -1
        current_index = 0

        for token in doc:
            if not token.is_punct:
                if token.i == target_token.i:
                    target_in_words_index = current_index
                words.append(token.text)
                current_index += 1

        if target_in_words_index == -1:
            #prevent memory leak
            text = sentence.rstrip('.?!')
            return text + " " + marker + "."

        #insert after the target word
        insert_pos = target_in_words_index + offset
        if insert_pos <= len(words):
            words.insert(insert_pos, marker)
            return " ".join(words) + "."
        else:
            return " ".join(words) + " " + marker + "."

    def rule_parity_negation(self, sentence):
        text = sentence.rstrip('.?!')
        words = text.split()

        if len(words) % 2 == 0:
            return "Not " + " ".join(words) + "."
        else:
            return " ".join(words) + " Not."


def process_input_file(input_file, output_prefix):
    factory = ImpossibleGrammarFactory()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return

    outputs = {
        'reversed': [],
        'fixed_distance': [],
        'parity_negation': []
    }

    for sentence in sentences:
        original = sentence

        reversed_sentence = factory.rule_reverse(original)
        fixed_distance_sentence = factory.rule_fixed_distance_insert(original, offset=2)
        parity_sentence = factory.rule_parity_negation(original)

        outputs['reversed'].append(reversed_sentence)
        outputs['fixed_distance'].append(fixed_distance_sentence)
        outputs['parity_negation'].append(parity_sentence)

    for rule_name, sentences_list in outputs.items():
        output_file = f"{output_prefix}_{rule_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences_list:
                f.write(f"Original: {original}\n")
                f.write(sentence + "\n\n")
        print(f"已生成文件: {output_file} (包含 {len(sentences_list)} 个句子)")

    print("\n数据完整性检查:")
    for rule_name in outputs:
        original_count = len(sentences)
        output_count = len(outputs[rule_name])
        if original_count == output_count:
            print(f"  ✓ {rule_name}: 输入{original_count}句，输出{output_count}句")
        else:
            print(f"  ✗ {rule_name}: 输入{original_count}句，输出{output_count}句 - 数据丢失！")


def verify_transformations(sample_sentences):
    """验证转换规则是否正常工作"""
    factory = ImpossibleGrammarFactory()
    
    print("转换规则验证:")
    print("=" * 60)
    
    for i, sentence in enumerate(sample_sentences, 1):
        print(f"\n原始句子 {i}: {sentence}")
        print(f"  反转规则: {factory.rule_reverse(sentence)}")
        print(f"  固定距离插入: {factory.rule_fixed_distance_insert(sentence, offset=2)}")
        print(f"  奇偶否定: {factory.rule_parity_negation(sentence)}")


if __name__ == "__main__":
    input_file = "input.txt"
    output_prefix = "impossible_output"

    print("=" * 60)
    print("impossible grammar generator")
    print("=" * 60)

    print(f"\ninput file is: {input_file}")
    process_input_file(input_file, output_prefix)

    print("\n" + "=" * 60)
    print("demonstration:")
    print("=" * 60)
    
    factory = ImpossibleGrammarFactory()
    sample_sentences = [
        "The black cat is eating a fresh fish.",
        "She can read books quickly.",
        "They have been waiting for hours.",
        "The students will study mathematics."
    ]
    
    verify_transformations(sample_sentences)
    
    print("\n" + "=" * 60)
    print("数据集统计:")
    print("=" * 60)
    
    for rule_name in ['reversed', 'fixed_distance', 'parity_negation']:
        output_file = f"{output_prefix}_{rule_name}.txt"
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                print(f"{rule_name:20} : {len(lines):5} 个句子")
        except FileNotFoundError:
            print(f"{rule_name:20} : 文件未找到")
