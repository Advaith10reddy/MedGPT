# 🧠 MedGPT: Medical LLM for Drug Information

[![Model on Hugging Face](https://img.shields.io/badge/HuggingFace-MedAlpaca-yellow?logo=huggingface)](https://huggingface.co/medalpaca/medalpaca-7b)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/medalpaca-medical-llm?style=social)](https://github.com/yourusername/medalpaca-medical-llm)

---
##Please Note, currently the model has only been trained on all the drugs available in the US Starting from Albhates A to CR
##More Incomming soon

## 🔍 Overview

**MedGPT** is a domain-specific fine-tuned **Large Language Model (LLM)** that can answer natural language queries related to **medical drugs** including:

- ✅ What is a drug used for?
- ✅ What is the generic name of a drug?
- ✅ What are the side effects (common and serious)?
- ✅ Full medical summary (uses, generics, and side effects)

**🚀 Base Model**: [`medalpaca/medalpaca-7b`](https://huggingface.co/medalpaca/medalpaca-7b)  
**🔧 Fine-Tuning Strategy**: LoRA (Low-Rank Adaptation)  
**💾 Quantization**: 4-bit using `bnb` (QLoRA compatible)  
**🌐 Hosting**: Ready for Hugging Face Spaces or Streamlit UI

---


## 🗂 Dataset

- ✅ Scraped from [**Drugs.com**](https://www.drugs.com) using [Playwright](https://playwright.dev/)
- ✅ Includes ~20k drug entries
- ✅ Format: Excel (`.xlsx`) and JSONL (`.jsonl`)
- ✅ Columns: `Drug Name`, `Generic Name`, `Side Effects`, `Uses`


## 🧠 Model Architecture

- 🔸 **Base**: [`medalpaca/medalpaca-7b`](https://huggingface.co/medalpaca/medalpaca-7b)
- 🔸 **PEFT**: LoRA via [PEFT library](https://github.com/huggingface/peft)
- 🔸 **Quantization**: 4-bit with `bnb_4bit` (BitsAndBytes) for memory efficiency
- 🔸 **Tokenizer**: Based on LLaMA tokenizer (`LlamaTokenizer`)



medical_bot/
├── data/
│   ├── excels/
│   │   └── drug_data_full.xlsx          # Full raw dataset
│   └── jsonl/
│       └── drug_data_full.jsonl         # JSONL for training
├── scripts/
│   ├── train_medalpaca.py               # Fine-tuning script
│   └── model/                           # Saved model and tokenizer
├── colab/
│   └── demo_colab.ipynb                 # Colab test notebook
├── README.md
└── requirements.txt

Each row is transformed into multiple QA pairs like:

```json
{
  "prompt": "What is Abecma used for?",
  "response": "Abecma is a CAR T-cell therapy used for treating relapsed or refractory multiple myeloma..."
}



