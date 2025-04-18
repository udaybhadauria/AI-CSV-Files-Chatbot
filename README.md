# AI Chatbot for CSV Files

🚀 **Interact with CSV Data Using AI**

A powerful chatbot that allows users to upload CSV files and query data using natural language or Pandas-based queries. With semantic search, users can get relevant results even without exact matches!

---

## 🔥 Features

✅ **Upload & Select CSV Files** – Easily upload or choose from existing files  
✅ **Pandas-Based Queries** – Use Python-style queries like `df.head()` or `df[df['Column'] == 'Value']`  
✅ **Semantic Search** – Find relevant data without exact wording  
✅ **Fast & Interactive UI** – Built with Gradio for a seamless experience  
✅ **Caching for Faster Queries** – Reduces redundant processing  
✅ **Handles Large CSVs** – Optimized to process big datasets  

---

## 🎯 How It Works

1️⃣ **Upload a CSV file** – or select from previously uploaded files  
2️⃣ **Ask questions** – Enter a Pandas-style query or a natural language question  
3️⃣ **Get instant results** – Data is displayed in a table format like in Excel  
4️⃣ **Switch between files** – Load different datasets dynamically  

---

## 🛠️ Tech Stack

- **Python** – Core logic  
- **FastAPI** – Backend for file uploads and API endpoints  
- **Gradio** – Interactive UI  
- **Pandas** – Data processing  
- **FAISS** – For semantic search  
- **SentenceTransformers** – Embedding model for AI-powered queries  

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/udaybhadauria/AI-CSV-Files-Chatbot.git
cd AI-Chatbot-CSV
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (For Mac/Linux)
venv\Scripts\activate  # (For Windows)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
uvicorn app:app --reload
```

The app will run locally at **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)**

---

## 🏗️ Upcoming Features

🚀 **SQL-like Query Support** – Write queries in SQL syntax instead of Pandas  
🚀 **Advanced Data Visualization** – Generate graphs from queries  
🚀 **Multi-File Querying** – Search across multiple datasets at once  
🚀 **Cloud Storage Integration** – Save and access CSVs from the cloud  

---

## 🤝 Contributing

Want to improve this project? Contributions are welcome!

1. **Fork** the repository  
2. Create a new branch  
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit  
   ```bash
   git commit -m "Added new feature"
   ```
4. Push the changes  
   ```bash
   git push origin feature-branch
   ```
5. Open a **Pull Request**  
