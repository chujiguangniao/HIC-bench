"""
Model API module for handling model interactions
"""

import json
import os
from datetime import datetime
from openai import OpenAI
from prompts import SCP_PROMPT, COT_PROMPT, RAG_PROMPT, RCP_PROMPT

class ModelAPI:
    def __init__(self, api_key, base_url, model_name, temperature=1.0, max_tokens=700):
        """
        初始化模型API
        
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

    def get_prompt(self, prompt_type, question, field=None, principle=None, knowledge=None):
        """
        获取提示词模板
        
        Args:
            prompt_type (str): 提示词类型
            question (str): 问题
            field (str): 领域
            principle (str): 原则
            knowledge (str): 知识库
            
        Returns:
            str: 格式化后的提示词
        """
        if prompt_type == 'scp':
            return SCP_PROMPT.format(field=field, question=question)
        elif prompt_type == 'cot':
            return COT_PROMPT.format(field=field, question=question)
        elif prompt_type == 'rag':
            return RAG_PROMPT.format(field=field, question=question, principle=principle)
        elif prompt_type == 'rcp':
            return RCP_PROMPT.format(field=field, question=question, principle=principle)
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    def generate_response(self, prompt):
        """
        生成模型回应
        
        Args:
            prompt (str): 提示词
            
        Returns:
            str: 模型回应
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        return response.choices[0].message.content

    def save_responses(self, question, response, prompt_type, output_file):
        """
        保存模型回应到JSON文件
        
        Args:
            question (str): 问题
            response (str): 模型回应
            prompt_type (str): 提示词类型
            output_file (str): 输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 准备数据
        data = {
            "model_name": self.model_name,
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response
        }

        # 如果文件存在，读取数据
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # 添加新数据
        existing_data.append(data)

        # 保存更新后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def process_question(self, question, prompt_type, field=None, principle=None, knowledge=None, output_file=None):
        """
        处理单个问题
        
        Args:
            question (str): 问题
            prompt_type (str): 提示词类型
            field (str): 领域
            principle (str): 原则
            knowledge (str): 知识库
            output_file (str): 输出文件路径
            
        Returns:
            str: 模型回应
        """
        # 获取提示词
        prompt = self.get_prompt(prompt_type, question, field, principle, knowledge)
        
        # 生成回应
        response = self.generate_response(prompt)
        
        # 保存回应到JSON
        if output_file:
            self.save_responses(question, response, prompt_type, output_file)
        
        return response 