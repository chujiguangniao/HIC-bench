import os
import numpy as np

# 定义文件夹列表
folders = ["normal", "COT", "RAG", "RCP", "T0-4"]
# 定义模型名称列表
models = ["chatgpt-4o-mini", "chatgpt-4o", "deepseek-v3", "deepseek-r1", "Qwen2.5-14b", "Qwen2.5-72b"]

# 遍历每个文件夹
for folder in folders:
    folder_path = os.path.join("result", folder)
    print(f"Processing folder: {folder}")

    # 遍历每个模型
    for model in models:
        file_name = f"{model}_evaluation.txt"  # 文件名格式为 {model}_evaluation.txt
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # 初始化列表来存储每个问题的 Originality 平均值
        originality_means = []

        # 打开文件并逐行读取
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file):
                # 去除行首尾空白字符
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 解析每一行数据
                parts = line.split()
                originality = int(parts[1])  # 提取 Originality 评分

                # 每10个回答为一组，存储 Originality 评分
                if line_num % 10 == 0:
                    question_scores = []
                question_scores.append(originality)

                # 每读取10行，计算一次平均值
                if (line_num + 1) % 10 == 0:
                    originality_mean = np.mean(question_scores)
                    originality_means.append(originality_mean)

        # 计算方差
        variance = np.var(originality_means)
        print(f"Variance of Originality scores for {model} in {folder}: {variance:.4f}")