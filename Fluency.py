#对模型进行Simcse测试，计算每个领域的平均相似度，得到Flu
import torch
import json
import itertools
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# 加载 SimCSE 模型
model_path_simcse = "./models/sup-simcse-bert-base-uncased"
tokenizer_simcse = AutoTokenizer.from_pretrained(model_path_simcse)
model_simcse = AutoModel.from_pretrained(model_path_simcse)

# 读取 JSON 文件
file_path = "HIC/result/scp/Qwen2.5-72b_answers.json"  # JSON 文件路径
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有答案
sentences = [item['answer'] for item in data]

num_questions = len(sentences)  # 计算问题的总数
num_domains = num_questions // 10  # 计算领域总数（每 10 个问题为一个领域）

# 计算每个领域的平均相似度
domain_similarities = []

for domain_idx in range(num_domains):
    domain_start = domain_idx * 100
    domain_end = (domain_idx + 1) * 100
    domain_sentences = sentences[domain_start:domain_end]

    question_similarities = []

    for i in range(10):
        group_sentences = domain_sentences[i * 10: (i + 1) * 10]
        pairs = list(itertools.combinations(group_sentences, 2))

        similarities = []
        for sent1, sent2 in pairs:
            inputs = tokenizer_simcse([sent1, sent2], padding=True, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                embeddings = model_simcse(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            similarity = 1 - cosine(embeddings[0].numpy(), embeddings[1].numpy())
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)
        question_similarities.append(avg_similarity)

    domain_avg_similarity = sum(question_similarities) / len(question_similarities)
    domain_similarities.append(domain_avg_similarity)

# 输出每个领域的平均相似度数组
print("Domain Similarities:", [round(sim, 4) for sim in domain_similarities])

# 输出总体平均相似度
total_avg_similarity = sum(domain_similarities) / len(domain_similarities)
print(f"Total Average Similarity: {total_avg_similarity:.4f}")


