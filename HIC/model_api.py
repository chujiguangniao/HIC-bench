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
            api_key (str): API密鑰
            base_url (str): API基礎URL
            model_name (str): 模型名稱
            temperature (float): 溫度參數
            max_tokens (int): 最大token數
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_prompt(self, prompt_type, question, field=None, principle=None, knowledge=None):
        """
        獲取提示詞模板
        
        Args:
            prompt_type (str): 提示詞類型
            question (str): 問題
            field (str): 領域
            principle (str): 原則
            knowledge (str): 知識庫
            
        Returns:
            str: 格式化後的提示詞
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
        生成模型回應
        
        Args:
            prompt (str): 提示詞
            
        Returns:
            str: 模型回應
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
        保存模型回應到JSON文件
        
        Args:
            question (str): 問題
            response (str): 模型回應
            prompt_type (str): 提示詞類型
            output_file (str): 輸出文件路徑
        """
        # 創建輸出目錄
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 準備要保存的數據
        data = {
            "model_name": self.model_name,
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response
        }

        # 如果文件存在，讀取現有數據
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # 添加新數據
        existing_data.append(data)

        # 保存更新後的數據
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def process_question(self, question, prompt_type, field=None, principle=None, knowledge=None, output_file=None):
        """
        處理單個問題
        
        Args:
            question (str): 問題
            prompt_type (str): 提示詞類型
            field (str): 領域
            principle (str): 原則
            knowledge (str): 知識庫
            output_file (str): 輸出文件路徑
            
        Returns:
            str: 模型回應
        """
        # 獲取提示詞
        prompt = self.get_prompt(prompt_type, question, field, principle, knowledge)
        
        # 生成回應
        response = self.generate_response(prompt)
        
        # 保存回應到JSON
        if output_file:
            self.save_responses(question, response, prompt_type, output_file)
        
        return response 