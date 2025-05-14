"""
评估器模組
"""

from openai import OpenAI
from prompts import EVALUATION_SYSTEM_PROMPT

class Evaluator:
    def __init__(self, api_key, base_url, model_name, temperature=0, max_tokens=200):
        """
        初始化评估器
        
        Args:
            api_key (str): API密钥
            base_url (str): API基础URL
            model_name (str): 模型名称
            temperature (float): 温度参数
            max_tokens (int): 最大token数
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def evaluate_answer(self, question, answer, answer_index):
        """
        评估单个答案
        
        Args:
            question (str): 原始问题
            answer (str): 要评估的答案
            answer_index (int): 答案索引
            
        Returns:
            str: 评估结果
        """
        print(f"Evaluating Answer {answer_index}...")
        
        user_prompt = f"[User Questions]:{question}\n[Answers to be evaluated]:{answer}"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        
        evaluation_result = response.choices[0].message.content.strip()
        print(evaluation_result)
        return evaluation_result
    
    def save_evaluation(self, evaluation_result, answer_index, output_file):
        """
        保存评估结果
        
        Args:
            evaluation_result (str): 评估结果
            answer_index (int): 答案索引
            output_file (str): 输出文件路径
        """
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{answer_index}:{evaluation_result}\n")
        print(f"Evaluation results appended to {output_file}")
        
    def process_evaluation(self, question, answer, answer_index, output_file):
        """
        处理单个答案的评估
        
        Args:
            question (str): 原始问题
            answer (str): 要评估的答案
            answer_index (int): 答案索引
            output_file (str): 输出文件路径
        """
        evaluation_result = self.evaluate_answer(question, answer, answer_index)
        self.save_evaluation(evaluation_result, answer_index, output_file) 