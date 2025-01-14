# **BIM AutoGen Dataset and LLM Fine-Tuning for making BIM LLM**

## **Overview**
This repository provides two simple tools for BIM-related machine learning tasks:
1. **BIM AutoGen Dataset**: Automatically generates QA datasets for fine-tuning LLMs from PDF documents.
2. **LLM Fine-Tuning with PEFT**: Fine-tunes a BIM-specific language model (SLM) using PEFT techniques on an 8GB GPU.

<div style="text-align: center;">
<img src="https://github.com/mac999/BIM_LLM/blob/main/doc/img1.JPG" height="250">
<img src="https://github.com/mac999/BIM_LLM/blob/main/doc/img3.JPG" height="250">
<img src="https://github.com/mac999/BIM_LLM/blob/main/doc/img5.JPG" height="150">
<img src="https://github.com/mac999/BIM_LLM/blob/main/doc/img2.JPG" height="150">
</div>

---

## **Features**
### **1. BIM AutoGen Dataset**
- Automates the generation of QA datasets from PDF files.
- Uses OpenAI API for question-answer generation and vagueness assessment.
- Processes and organizes datasets into JSON format for LLM training.

### **2. LLM Fine-Tuning with PEFT**
- Fine-tunes an LLM (e.g., Llama-3-8B) using lightweight **PEFT** (Parameter Efficient Fine-Tuning) techniques.
- Compatible with 8GB GPUs for cost-effective model training.
- Supports Hugging Face and W&B integrations for monitoring and sharing results.

---

## **Getting Started**

### **Prerequisites**
- Install Python 3.8 or above.
- Set up the required libraries using `pip install` (listed in the respective source codes).

---

### **1. BIM AutoGen Dataset**
#### **Usage**
1. **Install dependencies**:
   ```bash
   pip install os json PyPDF2 argparse re camelot fitz pdfminer.six openai tqdm
   ```
2. **Create OpenAI API Key**: Set up your API key for access.
3. **Prepare Input**: Add PDF documents to the `input` folder.
4. **Run Script**:
   ```bash
   python BIM_autogen_dataset.py --input ./input --output ./output
   ```
5. **Output**: QA datasets will be saved as JSON files in the `output` folder.

---

### **2. LLM Fine-Tuning with PEFT**
#### **Usage**
1. **Install dependencies**:
   ```bash
   pip install pandas torch wandb transformers huggingface_hub trl datasets peft PyPDF2 camelot-py pymupdf pdfminer.six openai tqdm
   ```
2. **Create API Keys**:
   - Hugging Face API Key.
   - Weights & Biases (W&B) API Key.
3. **Prepare Dataset**:
   - Place QA JSON files in the `dataset` folder (generated by BIM AutoGen Dataset).
4. **Run Script**:
   ```bash
   python BIM_LLM_finetuning.py
   ```
5. **Output**:
   - Fine-tuned model saved in the `output_finetuning_model` directory.
   - Training logs saved in `finetuning.log`.

---

## **References**
- **AutoGen Dataset**: Generates fine-tuning datasets with vagueness assessment.
- **Fine-Tuning**: Implements LoRA for efficient model tuning based on [Anyscale Blog](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2).
- **Research paper**: Training data and fine-tuning process for developing LLM-based BIM domain knowledge model based on [KCI portal](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003141449).

---

## **License**
This project is licensed under the MIT License. 

---

## **Contact**
For questions or collaboration opportunities:😊
- **Author**: Taewook Kang
- **Email**: laputa99999@gmail.com

--- 

