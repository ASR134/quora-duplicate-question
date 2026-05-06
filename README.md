# 🧠 SemantiQ – Duplicate Question Detection

A full-stack NLP application that detects whether two questions are **semantically duplicate** using Machine Learning.

Built with:
- ⚡ FastAPI (Backend API)
- 🎨 Streamlit (Frontend UI)
- 🤖 NLP + ML pipeline (feature engineering + model)

---

## 🚀 Live Demo
👉 Frontend: *(Add your Streamlit link here)*  
👉 API: https://quora-duplicate-question-zuq9.onrender.com

---

## ✨ Features

- 🔍 Detects semantic similarity between two questions  
- 📊 Returns prediction + confidence score  
- ⚡ FastAPI backend with trained ML model  
- 🎨 Clean Streamlit UI with analyzing animation  
- 🧠 Advanced NLP preprocessing + feature engineering  

---

## 🧠 How It Works

### 1. Text Preprocessing
- Lowercasing
- Removing punctuation & HTML
- Expanding contractions
- Tokenization (NLTK)
- Stemming (PorterStemmer)

---

### 2. Feature Engineering

#### 🔹 Basic Features
- Length of questions  
- Word count  
- Common word ratio  

#### 🔹 Advanced Features
- Token-based similarity
- Length-based similarity
- Longest common substring
- Fuzzy matching:
  - `QRatio`
  - `partial_ratio`
  - `token_sort_ratio`
  - `token_set_ratio`

---

### 3. Vectorization
- TF-IDF transformation for both questions  
- Features combined using `hstack`  

---

### 4. Model Prediction
- Trained ML model (`quora_model.pkl`)
- Outputs:
  - ✅ Duplicate  
  - ❌ Not Duplicate  
  - 📈 Confidence score  

---

## 🖥️ Frontend (Streamlit)

- Minimal and modern UI  
- Input fields for two questions  
- Animated **“Analyzing…” loader**  
- Displays:
  - Prediction
  - Confidence %

---

## 🔌 API Endpoints

### `GET /`
```json
{
  "message": "Quora Duplicate API is live"
}
```

---

### `GET /health`
```json
{
  "status": "ok"
}
```

---

### `POST /predict`

#### Request
```json
{
  "question1": "How to learn Python?",
  "question2": "Best way to start Python?"
}
```

#### Response
```json
{
  "prediction": "Duplicate",
  "confidence": 0.87
}
```

---

## ⚙️ Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/semantiq.git
cd semantiq
```

---

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run FastAPI server
```bash
uvicorn main:app --reload
```

---

### 4️⃣ Run Streamlit app
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
├── main.py              # FastAPI backend
├── app.py               # Streamlit frontend
├── model/
│   ├── quora_model.pkl
│   ├── quora_tf.pkl
│   ├── quora_scalar.pkl
├── requirements.txt
└── README.md
```

---

## 🧪 Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **ML/NLP:**  
  - Scikit-learn  
  - NLTK  
  - FuzzyWuzzy  
  - SciPy  
- **Deployment:** Render  

---

## ⚠️ Notes

- NLTK resources are downloaded at runtime  
- CORS is currently open (`*`) — restrict in production  
- First API call may be slow due to Render cold start  

---

## 📌 Future Improvements

- 🔥 Add deep learning model (BERT / Sentence Transformers)  
- 📊 Show similarity score visualization  
- 🌍 Multilingual support  
- ⚡ Cache model responses  

---

## 👤 Author

**Aman Rawat**