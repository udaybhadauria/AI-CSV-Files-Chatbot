import gradio as gr
import os
from fastapi import FastAPI, File, UploadFile, Query
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uvicorn
import threading

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI()

# Global variables
df = None
index = None
column_texts = None
current_file_name = None
current_columns = None
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache for query results
cache = {}

def get_uploaded_files():
    """Fetch the list of uploaded CSV files dynamically."""
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]

def load_and_process_csv(file_path):
    """Loads CSV and processes it for search."""
    global df, index, column_texts, current_file_name, current_columns

    df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df = df.fillna("N/A")
    column_texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    current_columns = ", ".join(df.columns)  # Get column names as a string

    # Generate embeddings
    row_embeddings = model.encode(column_texts, convert_to_numpy=True)

    # Build FAISS index
    dimension = row_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(row_embeddings)

    current_file_name = os.path.basename(file_path)
    return f"✅ CSV Loaded! {df.shape[0]} rows, {df.shape[1]} columns."

def chatbot_interface(uploaded_file, selected_file):
    """Handles file uploads and selection."""
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())  
        return load_and_process_csv(file_path)

    if selected_file:
        return load_and_process_csv(os.path.join(UPLOAD_FOLDER, selected_file))

    return "❌ No file uploaded or selected."

def update_ui(uploaded_file, selected_file):
    """Update UI when CSV is changed."""
    file_status = chatbot_interface(uploaded_file, selected_file)  # Returns status message
    uploaded_files = get_uploaded_files()  # Updated list of files
    return file_status, uploaded_files, current_file_name or "No CSV loaded", current_columns or "No columns available"

def handle_query(query):
    """Executes user queries and returns output in tabular format."""
    if not query or df is None:
        return "❌ No query entered or no CSV loaded."

    # Check if query is cached
    if query in cache:
        return cache[query]

    # Try executing as a Pandas query
    try:
        result = eval(query, {"df": df})
        
        if isinstance(result, pd.DataFrame):
            output = result.to_html(classes="table table-striped", index=False)
        else:
            output = str(result)

        cache[query] = output
        return output
    except Exception:
        pass

    # Perform semantic search if Pandas query fails
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, 1)  
    closest_match = df.iloc[indices[0][0]].to_frame().to_html(classes="table table-striped", index=False)

    cache[query] = closest_match
    return closest_match

def handle_manual_query(query):
    """Handles user-entered queries."""
    return handle_query(query)

def handle_predefined_query(predefined_query):
    """Handles predefined queries."""
    return handle_query(predefined_query)

# Gradio UI
with gr.Blocks() as gradio_ui:
    gr.Markdown("# 📊 AI-Powered CSV Chatbot - Uday Bhadauria")
    gr.Markdown("Upload or select a CSV file, then query it using Pandas commands or natural language.")

    with gr.Row():
        uploaded_file = gr.File(label="📤 Upload CSV File")
        selected_file = gr.Dropdown(label="📂 Select Previous File", choices=get_uploaded_files())

    load_btn = gr.Button("📥 Load File")
    file_name_display = gr.Textbox(label="Current CSV File", interactive=False)
    file_info = gr.Textbox(label="CSV File Info", interactive=False)
    column_names_display = gr.Textbox(label="CSV Data Column Names", interactive=False)

    load_btn.click(update_ui, inputs=[uploaded_file, selected_file], outputs=[file_info, selected_file, file_name_display, column_names_display])

    gr.Markdown("## 🔎 Query Options")

    with gr.Row():
        user_query = gr.Textbox(label="📝 Enter Manual Query", placeholder="e.g., df[df['Gender'] == 'Male'].shape[0]")
        query_btn = gr.Button("🔍 Run Query")

    manual_query_output = gr.HTML(label="Query Result")  # Tabular output

    with gr.Row():
        predefined_query = gr.Dropdown(label="📊 Predefined Queries", choices=[
            "",  
            "df.head()",  
            "df.tail()",  
            "df.shape",  
            "df.info()",
            "df.describe()",
            "df.columns",
            "df.dtypes",
            "df.nunique()",
            "df.isnull().sum()",  
        ])
        predefined_query_btn = gr.Button("📊 Run Predefined Query")

    predefined_query_output = gr.HTML(label="Query Result")  # Tabular output

    query_btn.click(handle_manual_query, inputs=[user_query], outputs=[manual_query_output])
    predefined_query_btn.click(handle_predefined_query, inputs=[predefined_query], outputs=[predefined_query_output])

def launch_gradio():
    """Launch Gradio UI in a separate thread."""
    gradio_ui.launch(share=True)

# Start Gradio in a separate thread
threading.Thread(target=launch_gradio).start()

# Serve FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
