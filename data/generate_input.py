import random

def generate_svo_sentences_with_auxiliary(count=10000):
    subjects_singular = [
        "The man", "The woman", "The boy", "The girl", "The student", 
        "The teacher", "The doctor", "The chef", "The artist", "The driver",
        "The cat", "The dog", "The bird", "The horse", "The rabbit",
        "The child", "The parent", "The friend", "The neighbor", "The worker",
        "The player", "The singer", "The dancer", "The writer", "The reader"
    ]
    
    subjects_plural = [
        "The men", "The women", "The boys", "The girls", "The students",
        "The teachers", "The doctors", "The chefs", "The artists", "The drivers",
        "The cats", "The dogs", "The birds", "The horses", "The rabbits",
        "The children", "The parents", "The friends", "The neighbors", "The workers",
        "The players", "The singers", "The dancers", "The writers", "The readers"
    ]
    
    pronouns = ["I", "You", "He", "She", "We", "They", "It"]
    
    auxiliaries = ["can", "will", "must", "should", "may", "might", "would", "could", "shall"]
    be_verbs = ["is", "am", "are", "was", "were", "has been", "have been", "had been"]
    have_verbs = ["has", "have", "had"]
    
    base_verbs = [
        "like", "love", "see", "hear", "know", "find", "take", "give",
        "make", "buy", "sell", "read", "write", "draw", "paint",
        "cook", "eat", "drink", "play", "watch", "study", "learn",
        "teach", "help", "call", "ask", "answer", "open", "close",
        "build", "create", "design", "develop", "understand", "explain",
        "remember", "forget", "enjoy", "hate", "prefer", "want", "need"
    ]
    
    past_participles = [
        "liked", "loved", "seen", "heard", "known", "found", "taken", "given",
        "made", "bought", "sold", "read", "written", "drawn", "painted",
        "cooked", "eaten", "drunk", "played", "watched", "studied", "learned",
        "taught", "helped", "called", "asked", "answered", "opened", "closed",
        "built", "created", "designed", "developed", "understood", "explained",
        "remembered", "forgotten", "enjoyed", "hated", "preferred", "wanted", "needed"
    ]
    
    present_participles = [
        "liking", "loving", "seeing", "hearing", "knowing", "finding", "taking", "giving",
        "making", "buying", "selling", "reading", "writing", "drawing", "painting",
        "cooking", "eating", "drinking", "playing", "watching", "studying", "learning",
        "teaching", "helping", "calling", "asking", "answering", "opening", "closing",
        "building", "creating", "designing", "developing", "understanding", "explaining",
        "remembering", "forgetting", "enjoying", "hating", "preferring", "wanting", "needing"
    ]
    
    objects = [
        "the book", "the pen", "the apple", "the car", "the house",
        "the ball", "the computer", "the phone", "the music", "the movie",
        "the food", "the water", "the coffee", "the tea", "the cake",
        "the picture", "the song", "the game", "the toy", "the flower",
        "the tree", "the sun", "the moon", "the sky", "the sea",
        "the dog", "the cat", "the bird", "the fish", "the horse",
        "the answer", "the question", "the problem", "the solution", "the idea",
        "the work", "the job", "the task", "the project", "the plan",
        "the painting", "the story", "the letter", "the message", "the news",
        "the lesson", "the class", "the school", "the garden", "the park"
    ]
    
    sentences = []
    
    for i in range(count):
        #choose the type of each sentence
        sentence_type = random.choice(["pronoun", "singular", "plural"])
        
        auxiliary_type = random.choice(["modal", "be_progressive", "be_passive", "have_perfect"])
        
        if sentence_type == "pronoun":
            subject = random.choice(pronouns)
        elif sentence_type == "singular":
            subject = random.choice(subjects_singular)
        else:  # plural
            subject = random.choice(subjects_plural)
        
        obj = random.choice(objects)
        verb = random.choice(base_verbs)
        verb_past = random.choice(past_participles)
        verb_present = random.choice(present_participles)
        
        def get_be_verb(subject):
            if subject in ["I"]:
                return "am"
            elif subject in ["He", "She", "It"] or (subject.startswith("The") and subject not in subjects_plural):
                return "is"
            elif subject in ["You", "We", "They"] or (subject.startswith("The") and subject in subjects_plural):
                return "are"
            return "is"  # 默认
        
        be_verb = get_be_verb(subject)
        
        def get_have_verb(subject):
            if subject in ["He", "She", "It"] or (subject.startswith("The") and subject not in subjects_plural):
                return "has"
            else:
                return "have"
        
        have_verb = get_have_verb(subject)
        
        if auxiliary_type == "modal":
            modal = random.choice(auxiliaries)
            sentence = f"{subject} {modal} {verb} {obj}."
        
        elif auxiliary_type == "be_progressive":
            if random.choice([True, False]):
                sentence = f"{subject} {be_verb} {verb_present} {obj}."
            else:
                # 过去时be动词
                past_be = "was" if be_verb == "is" or be_verb == "am" else "were"
                sentence = f"{subject} {past_be} {verb_present} {obj}."
        
        elif auxiliary_type == "be_passive":
            if random.choice([True, False]):
                sentence = f"{subject} {be_verb} {verb_past} {obj}."
            else:
                past_be = "was" if be_verb == "is" or be_verb == "am" else "were"
                sentence = f"{subject} {past_be} {verb_past} {obj}."
        
        else:  # have_perfect
            if random.choice([True, False]):
                sentence = f"{subject} {have_verb} {verb_past} {obj}."
            else:
                sentence = f"{subject} had {verb_past} {obj}."
        
        sentences.append(sentence)
    
    return sentences

def save_to_file(sentences, filename="input.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences, 1):
            f.write(f"Original: {sentence}\n")
            f.write(f"{sentence}\n\n")
    
    print(f"已生成 {len(sentences)} 个句子并保存到 {filename}")

sentences = generate_svo_sentences_with_auxiliary(100000)

save_to_file(sentences, "input.txt")

print("\n前10个句子示例：")
for i in range(10):
    print(f"{i+1}. {sentences[i]}")
