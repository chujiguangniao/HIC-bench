"""
提示模板配置文件
包含所有用于不同模型的提示模板
"""

# 基本提示模板
SCP_PROMPT = """Assume you are an expert in {field}. 
Please provide an answer to the following questions, not exceeding 70 tokens. Requirements:
1. Ensure feasibility by grounding in current scientific principles and technological trends;
2. Propose novel concepts or methods, avoiding unsupported speculation;
3. Ensure that the programme has value for the target area;
4. Maintain logical rigor without contradictions or vague statements.
Format: Plain text, no numbering or Markdown. There is a blank line between the answers
Question: {question}"""

# 思维链提示模板
COT_PROMPT = """Assume you are an expert in {field}. 
Please provide an answer to the following questions, not exceeding 70 tokens. Requirements:
1. Ensure feasibility by grounding in current scientific principles and technological trends;
2. Propose novel concepts or methods, avoiding unsupported speculation;
3. Ensure that the programme has value for the target area;
4. Maintain logical rigor without contradictions or vague statements.
5. Please think step by step before answering the question.
Format: Plain text, no numbering or Markdown. There is a blank line between the answers
Question: {question}"""

# RAG提示模板
RAG_PROMPT = """Assume you are an expert in {field}. 
Please provide an answer to the following questions, not exceeding 70 tokens. Requirements:
1. Ensure feasibility by grounding in current scientific principles and technological trends;
2. Propose novel concepts or methods, avoiding unsupported speculation;
3. Ensure that the programme has value for the target area;
4. Maintain logical rigor without contradictions or vague statements.
5. Answers should primarily rely on the provided Wikipedia principles; if information is insufficient, general scientific knowledge may be used, but speculation and contradictions must be avoided: {principle}.
Format: Plain text, no numbering or Markdown. There is a blank line between the answers
Question: {question}"""

# 宽松約束提示模板
RCP_PROMPT = """Assume you are an expert in {field}. 
Please provide an answer to the following questions, not exceeding 70 tokens. Requirements:
1. Propose novel concepts or methods, avoiding unsupported speculation;
2. Ensure that the program has potential value for the target area.
Format: Plain text, no numbering or Markdown. There is a blank line between the answers
Question: {question}"""

# 评估系統提示
EVALUATION_SYSTEM_PROMPT = """You are a rigorous evaluator specialized in assessing the scientific innovation quality of LLM-generated answers. 
Be highly critical and avoid giving high scores to generic or vague answers.
1. Evaluation Criteria (1-5 Scale):
- **Originality**:
1 = Common knowledge or widely known concepts with no novelty.
2 = Slight variation of known methods, minimal innovation.
3 = Moderate originality, combining existing ideas in a non-trivial way.
4 = Highly novel approach, limited prior research but plausible.
5 = Breakthrough idea with no direct precedent.
If the answer only combines **existing concepts** without a novel mechanism, the originality score must not exceed 3.
- **Feasibility**:
1 = Violates fundamental scientific principles.
2 = Theoretical possibility but lacks experimental validation.
3 = Feasible under specific assumptions but challenging to implement.
4 = Technically viable with existing technology.
5 = Readily implementable with minor adaptations of existing techniques.
- **Value**:
1 = No practical use or improvement over existing methods.
2 = Minor improvement, limited impact.
3 = Meaningful contribution with clear advantages.
4 = Significant improvement over current methods.
5 = Potential revolutionary impact on the field.
If the answer lacks technical details or fails to demonstrate security improvements, the value score must not exceed 3.
2. Hallucination Detection:
- If any of the following conditions are met, mark 'Hallucination: Yes':
- The answer does not align with the core requirements of the question.
- The answer deviates from reality.
- The answer contradicts established scientific principles.
- The answer provides irrelevant or tangential information without addressing the problem.
- The answer contains false information or made-up claims.
3. Scoring Rules:
- Generic or vague responses must receive lower scores: 'Originality <= 3' & 'Value <= 3'.
- Ensure a clear distinction between general answers and true innovations—avoid inflated scores.
4. Output format (strictly one line, no explanations):
'Originality: [1-5] Feasibility: [1-5] Value: [1-5] Hallucination: Yes/No'."""

