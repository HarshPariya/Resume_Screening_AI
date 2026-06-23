# 🚀 Resume Screening AI

An end-to-end NLP-powered Resume Screening System that automatically classifies uploaded resumes into professional job categories using Machine Learning and Natural Language Processing (NLP).

The system extracts text from PDF resumes, preprocesses the content, generates TF-IDF features, and predicts the most relevant job category using a trained XGBoost model.

---

## 📌 Project Overview

Recruiters often receive hundreds of resumes for a single job opening. Manually screening resumes is time-consuming and inefficient.

This project automates the resume screening process by analyzing resume content and predicting the candidate's professional domain.

The model was trained on **2,484 real-world PDF resumes** across **24 different job categories**.

---

## ✨ Features

- 📄 Upload Resume in PDF Format
- 🔍 Automatic Resume Text Extraction
- 🧹 NLP Text Preprocessing
- 📊 TF-IDF Feature Engineering
- 🤖 Machine Learning-Based Classification
- 🎯 Predicts Job Category
- 📈 Displays Prediction Confidence Score
- 💾 Trained Model Persistence
- 🌐 Streamlit Web Application
- ⚡ Real-Time Resume Screening

---

## 🏗️ Machine Learning Pipeline

```text
PDF Resume
     ↓
Text Extraction (pdfplumber)
     ↓
Text Cleaning & NLP Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Models
     ↓
Category Prediction
     ↓
Confidence Score
```

---

## 📂 Dataset Information

### Dataset Statistics

- Total Resumes: **2,484**
- Categories: **24**
- Format: **PDF**
- Converted into structured text for training

### Categories

- HR
- Aviation
- Consultant
- Designer
- Finance
- Construction
- Banking
- Public Relations
- Apparel
- Digital Media
- Chef
- Automobile
- Agriculture
- BPO
- Sales
- Arts
- Accountant
- Engineering
- Teacher
- Healthcare
- Information Technology
- Advocate
- Business Development
- Fitness

---

## 🧠 NLP Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove URLs
- Remove special characters
- Remove numbers
- Remove extra spaces
- Remove stopwords
- Text normalization

---

## ⚙️ Feature Engineering

### TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

---

## 🤖 Machine Learning Models Evaluated

| Model | Accuracy |
|---------|---------:|
| Logistic Regression | 66.80% |
| Tuned Logistic Regression | 69.01% |
| Linear SVM | 74.85% |
| Extra Trees Classifier | 75.25% |
| Random Forest Classifier | 77.26% |
| Multinomial Naive Bayes | 56.14% |
| 🏆 XGBoost Classifier | **80.89%** |

---

## 🏆 Final Model

### XGBoost Classifier

The XGBoost model achieved the highest performance among all evaluated models.

**Final Accuracy:** **80.89%**

### Why XGBoost?

- Highest accuracy among tested models
- Strong multi-class classification performance
- Handles high-dimensional TF-IDF features effectively
- Better generalization capability

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing & NLP

- Pandas
- NumPy
- NLTK
- pdfplumber

### Machine Learning

- Scikit-Learn
- XGBoost

### Deployment

- Streamlit

---

## 📁 Project Structure

```text
Resume_Screening_AI/
│
├── app.py
├── resume_model.pkl
├── tfidf.pkl
├── label_encoder.pkl
├── Resume_Screening_AI.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/HarshPariya/resume-screening-ai.git
```

### Navigate to Project Folder

```bash
cd resume-screening-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Results

- Processed **2,484 real-world resumes**
- Extracted text directly from PDF files
- Built a complete NLP pipeline
- Compared **7 Machine Learning Models**
- Achieved **80.89% Accuracy** using XGBoost
- Developed a scalable Resume Screening AI system

---

## 🔮 Future Enhancements

- ATS Resume Score Prediction
- Resume Ranking System
- Candidate Recommendation Engine
- Advanced Skill Extraction
- Job Description Matching
- Deep Learning & BERT Integration
- Recruiter Dashboard

---

## 👨‍💻 Author

### Harsh Pariya

Aspiring AI/ML Engineer passionate about:

- Machine Learning
- Natural Language Processing (NLP)
- Data Science
- Deep Learning
- Full-Stack AI Applications


---

