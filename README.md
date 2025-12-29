# Self-Harm Content Detection using NLP

This project focuses on detecting **self-harm related textual content** using Natural Language Processing (NLP) techniques. Multiple models were implemented and compared, including a traditional **Logistic Regression** classifier and a **BERT-based deep learning model**, with BERT achieving superior performance.

The goal is to explore how modern transformer-based models outperform classical approaches in sensitive text classification tasks.

---

## 📌 Project Overview

Online platforms often struggle to identify self-harm content early. This project aims to:

- Preprocess and clean raw text data
- Perform exploratory data analysis (EDA)
- Train and evaluate multiple NLP models
- Compare traditional ML models with transformer-based models
- Analyze performance differences and results

---

## 🧠 Models Used

### 1. Logistic Regression
- TF-IDF based feature extraction
- Baseline classical NLP model
- Faster training, lower computational cost

### 2. BERT (Bidirectional Encoder Representations from Transformers)
- Context-aware embeddings
- Fine-tuned for classification
- Significantly better performance on nuanced text

---




### File Descriptions

- **`data_cleaning.ipynb`**  
  Handles raw dataset cleaning, null removal, and text normalization.

- **`data_preprocessing_small.ipynb`**  
  Preprocessing pipeline applied to a reduced dataset for faster experimentation.

- **`eda.ipynb`**  
  Exploratory Data Analysis including class distribution, text length analysis, and insights.

- **`word2vec.ipynb`**  
  Experiments with Word2Vec embeddings for semantic representation.

- **`models_logit.ipynb`**  
  Logistic Regression model training and evaluation.

- **`infer_logit.ipynb`**  
  Inference pipeline for the trained Logistic Regression model.

- **`models_bert.ipynb`**  
  Fine-tuning and evaluation of the BERT model for self-harm detection.

---

## 📊 Results Summary

| Model               | Performance |
|--------------------|-------------|
| Logistic Regression | Baseline |
| BERT               | **Best results** |

The BERT model demonstrated superior understanding of contextual and sensitive language patterns compared to classical methods.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Google Colab

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/self-harm-detection.git

2. Open notebooks in Google Colab or Jupyter Notebook
3. Run the notebooks in the following order:

   1. `data_cleaning.ipynb`
   2. `data_preprocessing_small.ipynb`
   3. `eda.ipynb`
   4. `models_logit.ipynb`
   5. `models_bert.ipynb`
