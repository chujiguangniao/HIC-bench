# 对评分进行汇总
file_path = "result/SCP/Qwen2.5-72b_evaluation.txt"

# 初始化变量
num_domains = 10  # 领域数量
domain_size = 100  # 每个领域的回答数量d

# 用于存储总体数据
total_originality_sum = 0
total_feasibility_sum = 0
total_value_sum = 0
total_hallucination_no = 0
total_hallucination_yes = 0
total_intelligent_hallucination = 0  # 智能性幻觉数量
total_defective_hallucination = 0    # 缺陷性幻觉数量

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
        feasibility = int(parts[3])  # 提取 Feasibility 评分
        value = int(parts[5])        # 提取 Value 评分
        hallucination = parts[7]     # 提取 Hallucination 值

        # 累加总体评分
        total_originality_sum += originality
        total_feasibility_sum += feasibility
        total_value_sum += value

        # 统计 Hallucination 数量
        if hallucination.lower() == "no":
            total_hallucination_no += 1
        elif hallucination.lower() == "yes":
            total_hallucination_yes += 1

        # 判断是否为智能性幻觉
        if originality >= 4 and feasibility >= 3 and value >= 4:
            total_intelligent_hallucination += 1





# 计算总体平均值
total_lines = num_domains * domain_size
total_originality_avg = total_originality_sum / total_lines
total_feasibility_avg = total_feasibility_sum / total_lines
total_value_avg = total_value_sum / total_lines

# 输出总体结果
print("总体结果:")
print(f"  Originality 平均值: {total_originality_avg:.2f}")
print(f"  Feasibility 平均值: {total_feasibility_avg:.2f}")
print(f"  Value 平均值: {total_value_avg:.2f}")
print(f"  智能性幻觉数量: {total_intelligent_hallucination}")
print(f"  缺陷性幻觉数量: {total_hallucination_yes}")