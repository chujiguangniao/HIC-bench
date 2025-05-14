import pandas as pd
from openai import OpenAI
import time
import json
import yaml
import os
from typing import Dict, List, Any
from datetime import datetime

class DynamicPromptModel:
    def __init__(self, config_path: str = "config_dynamic.yaml"):
        # 获取当前文件的目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(
            api_key=self.config["api_settings"]["api_key"],
            base_url=self.config["api_settings"]["base_url"]
        )
        
        # 初始化动态提示词示例
        self.dynamic_prompt_examples = {"positive": "", "negative": ""}
        
        # 处理输出路径
        answers_dir = os.path.dirname(self.config["output_settings"]["answers_path"])
        eval_dir = os.path.dirname(self.config["output_settings"]["evaluation_path"])
        
        # 如果是相对路径，则相对于当前文件所在目录
        if not os.path.isabs(self.config["output_settings"]["answers_path"]):
            self.config["output_settings"]["answers_path"] = os.path.join(self.base_dir, self.config["output_settings"]["answers_path"])
        if not os.path.isabs(self.config["output_settings"]["evaluation_path"]):
            self.config["output_settings"]["evaluation_path"] = os.path.join(self.base_dir, self.config["output_settings"]["evaluation_path"])

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        # 如果是相对路径，则相对于当前文件所在目录
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.base_dir, config_path)
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_questions(self) -> tuple:
        """从 CDID 数据集加载问题和原理"""
        # 处理数据集路径
        dataset_path = self.config["data_settings"]["dataset_path"]
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.base_dir, dataset_path)
            
        df = pd.read_csv(dataset_path)
        return (
            df.iloc[:, self.config["data_settings"]["question_column"]].dropna().tolist(),
            df.iloc[:, self.config["data_settings"]["principle_column"]].dropna().tolist()
        )

    def _update_prompt(self) -> str:
        """更新动态提示词"""
        return (f"Assume you are an expert in the given field. "
                f"Please provide an answers to the following question,not exceeding 70 tokens. "
                f"Requirements:\n"
                f"1. Ensure feasibility based on scientific principles and technological trends.\n"
                f"2. Propose novel concepts or methods, avoiding unsupported speculation.\n"
                f"3. Ensure the answer has value for the target field.\n"
                f"4. Maintain logical rigor without contradictions or vague statements.\n"
                f"Format: Plain text, no numbering or Markdown. Each answer is separated by a blank line.\n"
                f"{self.dynamic_prompt_examples['positive']}"
                f"{self.dynamic_prompt_examples['negative']}")

    def _evaluate_answer(self, question: str, answer: str) -> str:
        """评估回答"""
        system_prompt = (
            "You are a rigorous evaluator specialized in assessing the scientific innovation quality of LLM-generated answers. "
            "Be highly critical and avoid giving high scores to generic or vague answers."
            "1. Evaluation Criteria (1-5 Scale):"
            "- **Originality**:"
            "1 = Common knowledge or widely known concepts with no novelty."
            "2 = Slight variation of known methods, minimal innovation."
            "3 = Moderate originality, combining existing ideas in a non-trivial way."
            "4 = Highly novel approach, limited prior research but plausible."
            "5 = Breakthrough idea with no direct precedent."
            "If the answer only combines **existing concepts** without a novel mechanism, the originality score must not exceed 3."
            "- **Feasibility**:"
            "1 = Violates fundamental scientific principles."
            "2 = Theoretical possibility but lacks experimental validation."
            "3 = Feasible under specific assumptions but challenging to implement."
            "4 = Technically viable with existing technology."
            "5 = Readily implementable with minor adaptations of existing techniques."
            "- **Value**:"
            "1 = No practical use or improvement over existing methods."
            "2 = Minor improvement, limited impact."
            "3 = Meaningful contribution with clear advantages."
            "4 = Significant improvement over current methods."
            "5 = Potential revolutionary impact on the field."
            "If the answer lacks technical details or fails to demonstrate security improvements, the value score must not exceed 3."
            "2. Hallucination Detection:"
            "- If any of the following conditions are met, mark 'Hallucination: Yes':"
            "- The answer does not align with the core requirements of the question."
            "- The answer deviates from reality."
            "- The answer contradicts established scientific principles."
            "- The answer provides irrelevant or tangential information without addressing the problem."
            "- The answer contains false information or  made-up claims."
            "3. Scoring Rules:"
            "- Generic or vague responses must receive lower scores: 'Originality <= 3' & 'Value <= 3'."
            "- Ensure a clear distinction between general answers and true innovations—avoid inflated scores."
            "4. Output format (strictly one line, no explanations):"
            "'Originality: [1-5] Feasibility: [1-5] Value: [1-5] Hallucination: Yes/No'."
        )

        user_prompt = (
            f"[User Questions]:{question}"
            f"[Answers to be evaluated]:{answer}"
        )

        response = self.client.chat.completions.create(
            model=self.config["evaluation_model_settings"]["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config["evaluation_model_settings"]["temperature"],
            max_tokens=self.config["evaluation_model_settings"]["max_tokens"],
        )
        return response.choices[0].message.content.strip()

    def _save_answers_to_json(self, answers: List[str], question_info: Dict[str, Any]) -> None:
        """将回答保存为 JSON 格式"""
        answer_data = {
            "question_id": question_info["global_index"] + 1,
            "field": question_info["field"],
            "question": question_info["question"],
            "timestamp": datetime.now().isoformat(),
            "answers": answers
        }

        # 读取现有数据或创建新的数据列表
        try:
            with open(self.config["output_settings"]["answers_path"], 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_data = []

        all_data.append(answer_data)

        # 保存更新后的数据
        with open(self.config["output_settings"]["answers_path"], 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

    def process_questions(self, start_question: int = 1) -> None:
        """处理所有问题并生成回答"""
        questions, principles = self._load_questions()

        start_global_index = start_question - 1
        start_field_index = start_global_index // 10
        start_question_index = start_global_index % 10

        for field_index in range(start_field_index, len(self.config["fields"])):
            field = self.config["fields"][field_index]
            start_index = field_index * 10
            end_index = start_index + 10
            field_questions = questions[start_index:end_index]
            field_principles = principles[start_index:end_index]

            if field_index == start_field_index:
                field_questions = field_questions[start_question_index:]
                field_principles = field_principles[start_question_index:]

            for i, (question, principle) in enumerate(zip(field_questions, field_principles),
                                                    start=start_question_index if field_index == start_field_index else 0):
                global_question_index = field_index * 10 + i

                print(f"处理问题 {global_question_index + 1}/{len(questions)} 在 {field} 领域...")

                # 生成回答
                prompt = self._update_prompt()
                response = self.client.chat.completions.create(
                    model=self.config["answer_model_settings"]["model_name"],
                    messages=[{"role": "user", "content": prompt + f"\nField: {field}\nQuestion: {question}"}],
                    temperature=self.config["answer_model_settings"]["temperature"],
                    max_tokens=self.config["answer_model_settings"]["max_tokens"]
                )

                answers = response.choices[0].message.content.strip().split("\n\n")
                
                # 保存回答
                self._save_answers_to_json(
                    answers,
                    {
                        "global_index": global_question_index,
                        "field": field,
                        "question": question
                    }
                )

                # 评估回答并更新动态提示词
                best_positive = {"score": 0, "text": ""}
                best_negative = ""

                with open(self.config["output_settings"]["evaluation_path"], "a", encoding="utf-8") as f:
                    for a_index, answer in enumerate(answers):
                        eval_result = self._evaluate_answer(question, answer)
                        print(eval_result)
                        f.write(f"{global_question_index * 10 + a_index + 1}: {eval_result}\n")

                        # 解析评估结果
                        parts = eval_result.split()
                        scores = {
                            parts[0].strip(":"): int(parts[1]),
                            parts[2].strip(":"): int(parts[3]),
                            parts[4].strip(":"): int(parts[5])
                        }
                        hallucination = parts[7].strip().lower() == "yes"

                        # 更新动态提示词示例
                        total_score = sum(scores.values())
                        if scores["Originality"] >= 4 and scores["Feasibility"] >= 3 and scores["Value"] >= 4:
                            if total_score > best_positive["score"]:
                                best_positive = {"score": total_score, "text": f"Positive Example:\n{answer}\n"}

                        if hallucination:
                            best_negative = f"Negative Example (Hallucination):\n{answer}\n"

                    if best_positive["text"]:
                        self.dynamic_prompt_examples["positive"] = best_positive["text"]
                    if best_negative:
                        self.dynamic_prompt_examples["negative"] = best_negative

                time.sleep(1)

        print("处理完成。")

if __name__ == "__main__":
    # 创建模型实例并运行
    model = DynamicPromptModel()
    model.process_questions(start_question=1) 