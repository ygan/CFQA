#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public evaluation entrypoint for FinQA answer scoring."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

QUESTION_KEY = "问题"
GOLD_ANSWER_KEY = "答案"
ANALYSIS_KEY = "LLM_Evaluator_Analysis"
JUDGE_KEY = "LLM_Evaluator_Result"
DEFAULT_ANSWER_PATHS = [
    "llm_answer.answer",
    "llm_answer",
    "chat",
    "answer",
]


def create_common_prompt(question, correct_answer, student_answer):
    """
    生成一个用于判断回答1和回答2关键信息是否一致或包含的提示语。
    返回值以 JSON 格式返回, 只包含一个键: "prompt"。
    """
    merged_prompt = f"""请根据问题以及正确答案，逐步分析学生答案是否正确。
当学生答案包含正确答案的大部分信息时，学生答案正确。例如，正确答案中存在三点信息，而学生答案中有2点内容与正确答案一致，此时，正确答案超过一半的信息都出现在学生答案中，可以认为学生答案正确。
具体公司名称和年份不需要学生答出，允许学生忽略这两个信息，即关键信息中不包含公司名称和年份。
如果正确答案包含数字，只要学生给出误差在5%以内的相似数字，则可以认为正确。
学生答案的表述必须与正确答案在逻辑和结构上保持一致，不能仅仅表达相似的概念或核心点。
不需要关注信息来源，比如正确答案中提到根据表格得出结论，而学生答案中未提到表格而直接给出正确结论，那么我们也认为学生答案正确。

正确答案：
{correct_answer}

学生答案：
{student_answer}

分析学生答案是否正确，并以Json格式返回，Json格式如下：
{{
    "analysis": "分析正确答案的关键信息及其数量，然后与学生答案的关键信息及其数量进行比较，验证数字是否接近，学生答案是否包含超过一半的正确信息",
    "correct_or_not": True or False
}}
"""
    return merged_prompt


def create_non_exist_prompt(question, correct_answer, student_answer):
    merged_prompt = f"""已知问题的答案无法在财报中直接找到答案，请分析学生是否发现了这一点，如果学生发现了，则学生答案正确。即只要学生答案表达出类似财报无法找到相关的信息，便算正确。学生说，答财报没有直接回答这个问题，但间接回答了这个问题也算正确。但是如果学生没有发现财报中找不到这个问题的答案，而只是回答一些不相干的内容，则学生答案不正确。

    问题: {question}

    学生答案：
    {student_answer}

    分析学生答案是否正确，并以Json格式返回，Json格式如下：
    {{
        "analysis": "分析学生答案是否正确",
        "correct_or_not": True or False
    }}
    """
    return merged_prompt



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted answers against FinQA gold answers.")
    parser.add_argument("--input", help="Legacy mode: a single JSON file containing both predictions and gold.")
    parser.add_argument("--predictions-file", default="examples/DeepSeek.json", help="Path to the prediction JSON file.")
    parser.add_argument("--gold-file",default="dataset/split_by_company/split_by_company_test.json", help="Path to the gold/reference JSON file.")
    parser.add_argument("--output", help="Path to the annotated output JSON.")
    parser.add_argument("--summary-output", help="Path to the summary JSON.")
    parser.add_argument(
        "--answer-key",
        default="llm_answer.answer",
        help="Prediction field to evaluate. Use dotted paths such as llm_answer.answer or chat.",
    )
    parser.add_argument("--id-key", default="id", help="Primary key used to align predictions with gold.")
    parser.add_argument("--question-key", default=QUESTION_KEY, help="Question field name.")
    parser.add_argument("--gold-key", default=GOLD_ANSWER_KEY, help="Gold answer field name.")
    parser.add_argument("--analysis-key", default=ANALYSIS_KEY, help="Field name for evaluation analysis.")
    parser.add_argument("--judge-key", default=JUDGE_KEY, help="Field name for evaluation label.")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai")
    parser.add_argument("--model", default="gpt-4o-2024-11-20", help="Judge model name.")
    parser.add_argument("--cache", default="log/gpt-4o.pkl", help="Cache file path.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N records.")
    return parser.parse_args()


def parse_path(path_text: str) -> List[str]:
    return [part for part in path_text.split(".") if part]


def has_nested(record: dict, path: Sequence[str]) -> bool:
    current = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def get_nested(record: dict, path: Sequence[str]):
    current = record
    for key in path:
        if not isinstance(current, dict):
            raise KeyError(".".join(path))
        current = current[key]
    return current


def normalize_answer(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "answer" in value:
            return normalize_answer(value["answer"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ".join(normalize_answer(item) for item in value if normalize_answer(item))
    return str(value).strip()


def detect_answer_path(records: Iterable[dict], answer_key: str = None) -> List[str]:
    candidates = [answer_key] if answer_key else DEFAULT_ANSWER_PATHS
    sample_records = list(records)
    for candidate in candidates:
        path = parse_path(candidate)
        for record in sample_records:
            if has_nested(record, path) and normalize_answer(get_nested(record, path)):
                return path
    joined = ", ".join(DEFAULT_ANSWER_PATHS)
    raise KeyError(f"Cannot find a usable prediction field. Try --answer-key explicitly. Default candidates: {joined}")


def evaluate_record(llm, question: str, gold_answer: str, prediction: str):
    if gold_answer == "无法在年报中找到相关信息":
        prompt = create_non_exist_prompt(question, gold_answer, prediction)
    else:
        prompt = create_common_prompt(question, gold_answer, prediction)

    content, _ = llm.request(prompt, None, json_format=True)
    if not content:
        return 0, None

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return 0, None

    return (1 if result.get("correct_or_not") else 0), result.get("analysis")


def common_prompt_eval(llm, record, question_key: str, answer1_path: Sequence[str], answer2_key: str):
    if len(answer1_path) > 1:
        answer1_parent = get_nested(record, answer1_path[:-1])
        if isinstance(answer1_parent, dict) and answer1_path[-1] in answer1_parent:
            answer1 = answer1_parent[answer1_path[-1]]
        else:
            answer1 = get_nested(record, answer1_path[:-1])
    else:
        answer1 = get_nested(record, answer1_path)

    prompt = create_common_prompt(record[question_key], record[answer2_key], answer1)
    content, _ = llm.request(prompt, None, json_format=True)
    if content:
        try:
            output = json.loads(content)
        except Exception:
            return -1, None
    else:
        return -1, None

    if "correct_or_not" not in output:
        return 1, None
    elif output["correct_or_not"]:
        return 1, output["analysis"]
    else:
        return 0, output["analysis"]


def non_exist_prompt_eval(llm, record, question_key: str, answer1_path: Sequence[str], answer2_key: str):
    if len(answer1_path) > 1:
        answer1 = get_nested(record, answer1_path)
    else:
        answer1 = get_nested(record, answer1_path)

    prompt = create_non_exist_prompt(record[question_key], record[answer2_key], answer1)
    content, _ = llm.request(prompt, None, json_format=True)
    if content:
        output = json.loads(content)
    else:
        return -1, None

    if output["correct_or_not"]:
        return 1, output["analysis"]
    else:
        return 0, output["analysis"]


def evaluate_record_original_style(llm, record, question_key: str, answer1_path: Sequence[str], answer2_key: str):
    if record[answer2_key] == "无法在年报中找到相关信息":
        return non_exist_prompt_eval(llm, record, question_key, answer1_path, answer2_key)
    else:
        return common_prompt_eval(llm, record, question_key, answer1_path, answer2_key)


def build_llm(provider: str, model: str, cache_path: str):
    from utils.llm import ChatGPTBatch,ChatGPT, DeepSeek

    if provider == "deepseek":
        return DeepSeek(model, cache=cache_path)
    return ChatGPT(model, cache=cache_path)


def default_output_paths(input_path: Path):
    output_path = input_path.with_name(f"{input_path.stem}.eval.json")
    summary_path = input_path.with_name(f"{input_path.stem}.summary.json")
    return output_path, summary_path


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def build_gold_lookup(records: List[dict], id_key: str) -> dict:
    lookup = {}
    for record in records:
        if id_key not in record:
            raise KeyError(f"Gold record is missing id field `{id_key}`.")
        lookup[record[id_key]] = record
    return lookup


def main() -> None:
    args = parse_args()
    if args.input:
        input_path = Path(args.input)
        records = load_json_list(input_path)
        gold_lookup = None
        predictions_path = input_path
        gold_path = None
    else:
        if not args.predictions_file or not args.gold_file:
            raise ValueError("Use either `--input`, or provide both `--predictions-file` and `--gold-file`.")
        predictions_path = Path(args.predictions_file)
        gold_path = Path(args.gold_file)
        records = load_json_list(predictions_path)
        gold_records = load_json_list(gold_path)
        gold_lookup = build_gold_lookup(gold_records, args.id_key)
        input_path = predictions_path

    records = records[: args.limit] if args.limit else records
    answer_path = detect_answer_path(records[: min(len(records), 50)], args.answer_key)
    llm = build_llm(args.provider, args.model, args.cache)

    gpt_correct = 0
    missing_prediction_count = 0
    annotated_records = []

    for index, record in enumerate(records, start=1):
        if gold_lookup is None:
            merged_record = dict(record)
        else:
            if args.id_key not in record:
                raise KeyError(f"Prediction record is missing id field `{args.id_key}`.")
            record_id = record[args.id_key]
            if record_id not in gold_lookup:
                raise KeyError(f"Prediction id `{record_id}` was not found in the gold file.")
            gold_record = gold_lookup[record_id]
            merged_record = dict(gold_record)
            for key, value in record.items():
                merged_record[key] = value

        if not has_nested(merged_record, answer_path):
            result = 0
            analysis = "Prediction field is missing."
            missing_prediction_count += 1
        else:
            result, analysis = evaluate_record_original_style(
                llm,
                merged_record,
                args.question_key,
                answer_path,
                args.gold_key,
            )

        merged_record[args.analysis_key] = analysis
        merged_record[args.judge_key] = result
        annotated_records.append(merged_record)

        auto_label = 1 if result else 0
        gpt_correct += 1 if auto_label else 0

        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} records")

    accuracy = gpt_correct / len(records) if records else 0.0
    summary = {
        "predictions_file": str(predictions_path),
        "gold_file": str(gold_path) if gold_path else None,
        "records": len(records),
        "provider": args.provider,
        "model": args.model,
        "answer_key": ".".join(answer_path),
        "gpt_eval_accuracy": accuracy,
        "missing_prediction_count": missing_prediction_count,
    }

    default_output, default_summary = default_output_paths(input_path)
    output_path = Path(args.output) if args.output else default_output
    summary_path = Path(args.summary_output) if args.summary_output else default_summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotated_records, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"gpt eval accuracy {accuracy}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Annotated file saved to: {output_path}")
    print(f"Summary file saved to: {summary_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
