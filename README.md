# Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations

![image](https://github.com/user-attachments/assets/cd652394-c84d-46cd-8e4f-50e6c134c605)


HIC-Bench is an innovative evaluation framework for studying the dual nature of hallucinations in Large Language Models (LLMs). By categorizing hallucinations into Intelligent Hallucinations (IH) and Defective Hallucinations (DH), we examine their interplay through LLM creativity across open-ended innovation tasks in ten scientific domains.

This repository contains the datasets and evaluation scripts for the paper **Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations**. HIC-Bench is a set of benchmark tasks and datasets crafted to evaluate the classification and assessment of hallucinations in Large Language Models (LLMs), classifying their intelligent and defective aspects from the perspective of creativity in open-ended scientific tasks across various domains.

## Framework Overview

The system consists of three core modules:
- **HIC (IH&DH)**: Evaluates LLM hallucinations in open-ended innovation tasks
- **DHP (Dynamic Hallucination Prevention)**: Dynamically assesses and optimizes hallucination distributions
- **Other Tools**: Including Fluency and Flexibility evaluation metrics

### Key Features

- 🧠 Intelligent Hallucination Detection: Identifies and evaluates valuable creative content
- 📊 Multi-dimensional Assessment: Covers originality, feasibility, and value
- 🔄 Dynamic Prompt Optimization: Automatically adjusts prompting strategies based on evaluation results
- 📈 Quantitative Analysis: Supports distribution analysis of IH and DH

## 1. HIC Module Guide

The HIC module is the core component for evaluating LLM performance in scientific innovation tasks.

Dataset details can be found at huggingface: [https://huggingface.co/datasets/chujiguangniao/HIC-Bench](https://huggingface.co/datasets/chujiguangniao/HIC-Bench)

### 1.1 Configuration

Configure parameters in `HIC/config.yaml`:

### 1.2 Running Evaluations

Please fill in your own api_key in the yaml file first:

```
model_api_key: your_api_key
```

Basic execution:
```bash
cd HIC
python main.py --config config.yaml
```

Custom parameters:
```bash
python main.py --model_name "gpt-4o" --prompt_type "scp" --start_question 1
```

### 1.3 Output Analysis

The system generates two key file types:
- `{model_name}_scp_responses.json`: Model responses with creativity analysis
- `{eval_model_name}_scp_evaluations.json`: IH/DH classification and evaluation metrics

## 2. DHP Module Guide

The DHP module optimizes prompting strategies to balance creativity and accuracy.

### 2.1 Running Analysis

```bash
cd HIC/DHP
python dhp.py
```

## 3. Auxiliary Evaluation Tools

### 3.1 Fluency Assessment

Assess the ability to generate multiple different responses:

Please download the `simcse` model file first [sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)

```bash
python Fluency.py
```

### 3.2 Flexibility Assessment

Analyzes conceptual diversity and cognitive flexibility:
```bash
python Flexibility.py
```



## Important Notes

1. API Security: Ensure secure storage of all API keys
2. We generate and evaluate ten responses for each piece of data.
3. Result Interpretation: Pay attention to IH and DH characteristics

