"""
主程序模組
"""

import os
import argparse
import yaml
import pandas as pd
from model_api import ModelAPI
from evaluator import Evaluator

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路徑
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM Creativity Evaluation System')
    
    # 模型API参数
    parser.add_argument('--model_api_key', type=str,
                      help='Model API Key')
    parser.add_argument('--model_base_url', type=str,
                      help='Model API Base URL')
    parser.add_argument('--model_name', type=str,
                      help='Model Name')
    parser.add_argument('--model_temperature', type=float,
                      help='Model Temperature')
    parser.add_argument('--model_max_tokens', type=int,
                      help='Model Max Tokens')
    
    # 评估器参数
    parser.add_argument('--eval_api_key', type=str,
                      help='Evaluator API Key')
    parser.add_argument('--eval_base_url', type=str,
                      help='Evaluator API Base URL')
    parser.add_argument('--eval_model_name', type=str,
                      help='Evaluator Model Name')
    parser.add_argument('--eval_temperature', type=float,
                      help='Evaluator Temperature')
    parser.add_argument('--eval_max_tokens', type=int,
                      help='Evaluator Max Tokens')
    
    # 提示词类型
    parser.add_argument('--prompt_type', type=str,
                      choices=['scp', 'cot', 'rag', 'rcp'],
                      help='Prompt Type')
    
    # 文件路径
    parser.add_argument('--dataset_path', type=str,
                      help='Dataset Path')
    parser.add_argument('--output_dir', type=str,
                      help='Output Directory')
    
    # 起始问题
    parser.add_argument('--start_question', type=int,
                      help='Start Question Number')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration File Path')
    
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)
    
    # 使用命令行参数覆盖配置文件
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    # 创建輸出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 初始化模型API
    model_api = ModelAPI(
        api_key=config['model_api_key'],
        base_url=config['model_base_url'],
        model_name=config['model_name'],
        temperature=config['model_temperature'],
        max_tokens=config['model_max_tokens']
    )
    
    # 初始化评估器
    evaluator = Evaluator(
        api_key=config['eval_api_key'],
        base_url=config['eval_base_url'],
        model_name=config['eval_model_name'],
        temperature=config['eval_temperature'],
        max_tokens=config['eval_max_tokens']
    )

    # 读取数据集
    df = pd.read_csv(config['dataset_path'])
    
    # 处理每个问题
    for index, row in df.iterrows():
        question_number = index + 1
        if question_number < config['start_question']:
            continue
            
        print(f"\nProcessing Question {question_number}:")
        print(f"Field: {row['Field']}")
        print(f"Question: {row['Question']}")
        print(f"Principle: {row['Principle']}")
        
        # 生成答案并保存到JSON
        response = model_api.process_question(
            question=row['Question'],
            prompt_type=config['prompt_type'],
            field=row['Field'],
            principle=row['Principle'],
            knowledge=row['Knowledge Base'],
            output_file=os.path.join(
                config['output_dir'],
                f"{config['model_name']}_{config['prompt_type']}_responses.json"
            )
        )
        
        print(f"\nModel Response:\n{response}")
        
        # 评估答案并保存到TXT
        evaluator.process_evaluation(
            question=row['Question'],
            answer=response,
            answer_index=question_number,
            output_file=os.path.join(
                config['output_dir'],
                f"{config['eval_model_name']}_{config['prompt_type']}_evaluations.txt"
            )
        )

if __name__ == '__main__':
    main()