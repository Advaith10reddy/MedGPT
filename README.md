# 🧠 Med_AI: Medical LLM for Drug Information

[![Model on Hugging Face](https://img.shields.io/badge/HuggingFace-MedAlpaca-yellow?logo=huggingface)](https://huggingface.co/medalpaca/medalpaca-7b)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/medalpaca-medical-llm?style=social)](https://github.com/yourusername/medalpaca-medical-llm)

---

## 🔍 Overview

**MedAlpaca** is a domain-specific fine-tuned **Large Language Model (LLM)** that can answer natural language queries related to **medical drugs** including:

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

Each row is transformed into multiple QA pairs like:

```json
{
  "prompt": "What is Abecma used for?",
  "response": "Abecma is a CAR T-cell therapy used for treating relapsed or refractory multiple myeloma..."
}
