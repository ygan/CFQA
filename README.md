<div align="right">
  <a href="#english-version">
    <img alt="Switch to English" src="https://img.shields.io/badge/English-Version-black?style=for-the-badge">
  </a>
</div>

# CFQA

中文金融问答基准数据集与评测代码。

## 简介

CFQA 是一个面向中文上市公司年报问答任务的基准数据集。

由于原始财报、问题与标准答案均为中文，因此数据集内容以中文为主。本仓库提供：

- 中文数据集划分文件
- 公开评测代码
- 示例预测结果与参考评测分数

## 数据说明

数据集位于 `dataset/` 目录下，提供按公司划分和按年份划分的训练、验证、测试集：

- `dataset/split_by_company/`
- `dataset/split_by_year/`

每条样本通常包含如下字段：

- `股票代码`
- `公司`
- `问题`
- `答案`
- `答案出自`
- `id`

此外，用户问题中通常可以提取出其对应的年报年份。根据样本中的公司信息和问题中涉及的年份，可以定位到相应的上市公司年报。对应年报文件可从公开网站下载，例如交易所官网、巨潮资讯网等公开披露渠道。

字段 `答案出自` 表示答案对应的财报 PDF 页码索引，起始值为 1。该索引对应的是 PDF 文档页序，而不是财报页面底部印刷的页码。

## 安装

推荐使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

如果你使用基于 OpenAI 或 Deepseek 的评测器，请先在环境变量中配置相应的 API 凭证。

## 评测方式

运行公开评测脚本：

使用 OpenAI 评测器：

```bash
python evaluation.py \
  --predictions-file examples/DeepSeek.json \
  --gold-file dataset/split_by_company/split_by_company_test.json \
  --answer-key llm_answer.answer \
  --model gpt-4o-2024-11-20 \
  --cache log/gpt-4o.pkl
```

如果使用 DeepSeek 评测器，则运行：

```bash
python evaluation.py \
  --predictions-file examples/DeepSeek.json \
  --gold-file dataset/split_by_company/split_by_company_test.json \
  --answer-key llm_answer.answer \
  --provider deepseek \
  --model deepseek-chat \
  --cache log/deepseek-chat.pkl
```

使用上述命令时，理论上gpt-4o-2024-11-20评估模型应得到接近以下结果：

```text
0.6419047619047619
```


## 输出说明

评测脚本通常会输出：

- 带有逐条评测结果的 `*.eval.json`
- 汇总指标文件 `*.summary.json`

汇总文件中的 `gpt_eval_accuracy` 是主要报告指标。


## 引用

如果你在研究中使用了 CFQA，请引用：

```bibtex
@article{Gan2025,
  author = {Gan, Yujian and Tao, Yiyi and Mo, Jiawang and Huang, Xianzheng and Li, Yiwen and Wang, Kexin and Cai, Yi and Liang, Lu and Xiong, Shuzhen and Ke, Qi and Zheng, Hua and Hu, Xiaochun},
  title = {Addressing investor concerns: a Chinese financial question-answering benchmark with LLM-based evaluation},
  journal = {EPJ Data Science},
  year = {2025},
  volume = {15},
  number = {1},
  pages = {6},
  doi = {10.1140/epjds/s13688-025-00601-6},
  url = {https://doi.org/10.1140/epjds/s13688-025-00601-6}
}
```

## 许可协议

本项目遵循 [`LICENSE`](LICENSE) 中的许可协议。

---

<div align="right">
  <a href="#cfqa">
    <img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-%E7%89%88%E6%9C%AC-black?style=for-the-badge">
  </a>
</div>

## English Version

CFQA is a Chinese financial question-answering benchmark and evaluation toolkit.

## Overview

CFQA is a benchmark for question answering over annual reports of Chinese listed companies.

Because the source reports, questions, and gold answers are all in Chinese, the dataset content is primarily Chinese. This repository provides:

- Chinese dataset splits
- Public evaluation code
- Example predictions and reference evaluation results

## Dataset

The dataset is released under the `dataset/` directory, with splits by company and by year:

- `dataset/split_by_company/`
- `dataset/split_by_year/`

Each record typically contains fields such as:

- `股票代码`
- `公司`
- `问题`
- `答案`
- `答案出自`
- `id`

In addition, the annual-report year relevant to a question can usually be inferred from the user question itself. Combined with the company information in each sample, this makes it possible to locate the corresponding annual report. The source annual reports can be downloaded from public websites, such as stock exchange websites and CNINFO-like public disclosure platforms.

The `答案出自` field stores the page index in the annual-report PDF where the answer is grounded. The index starts from 1. It refers to the PDF document page order, not the printed page number shown at the bottom of the report page.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

If you use the OpenAI or Deepseek based evaluator, make sure the corresponding API credentials are configured in your environment.

## Evaluation

Run the public evaluator with:

Using the OpenAI evaluator:

```bash
python evaluation.py \
  --predictions-file examples/DeepSeek.json \
  --gold-file dataset/split_by_company/split_by_company_test.json \
  --answer-key llm_answer.answer \
  --model gpt-4o-2024-11-20 \
  --cache log/gpt-4o.pkl
```

If you want to evaluate with DeepSeek instead, run:

```bash
python evaluation.py \
  --predictions-file examples/DeepSeek.json \
  --gold-file dataset/split_by_company/split_by_company_test.json \
  --answer-key llm_answer.answer \
  --provider deepseek \
  --model deepseek-chat \
  --cache log/deepseek-chat.pkl
```

The command with gpt-4o-2024-11-20 should produce a result close to:

```text
0.6419047619047619
```

## Output

The evaluator typically writes:

- an annotated evaluation file, usually `*.eval.json`
- a summary file, usually `*.summary.json`

The `gpt_eval_accuracy` field in the summary JSON is the main reported metric.

## Citation

If you use CFQA in your research, please cite:

```bibtex
@article{Gan2025,
  author = {Gan, Yujian and Tao, Yiyi and Mo, Jiawang and Huang, Xianzheng and Li, Yiwen and Wang, Kexin and Cai, Yi and Liang, Lu and Xiong, Shuzhen and Ke, Qi and Zheng, Hua and Hu, Xiaochun},
  title = {Addressing investor concerns: a Chinese financial question-answering benchmark with LLM-based evaluation},
  journal = {EPJ Data Science},
  year = {2025},
  volume = {15},
  number = {1},
  pages = {6},
  doi = {10.1140/epjds/s13688-025-00601-6},
  url = {https://doi.org/10.1140/epjds/s13688-025-00601-6}
}
```

## License

This project is released under [`LICENSE`](LICENSE).
