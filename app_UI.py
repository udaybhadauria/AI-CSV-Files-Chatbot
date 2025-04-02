import os
import io
import threading
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import gradio as gr
import uvicorn
import matplotlib
from sentence_transformers import SentenceTransformer
import asyncio
from fastapi.responses import StreamingResponse
import io

# Set Matplotlib backend to 'Agg' (non-interactive)
matplotlib.use('Agg')

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df = None
index = None
current_file_name = None
current_columns = None
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLite database connection (in-memory)
conn = sqlite3.connect(":memory:", check_same_thread=False)

# Lock to ensure thread safety
lock = threading.Lock()

def get_uploaded_files():
    """Fetch the list of uploaded CSV files dynamically."""
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles CSV file uploads via API."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/")
def list_files():
    """Lists all uploaded CSV files."""
    return {"files": get_uploaded_files()}

@app.get("/load/{file_name}")
def load_csv(file_name: str):
    """Loads a CSV file and processes it."""
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in {UPLOAD_FOLDER}")

    load_and_process_csv(file_path)

    return {"message": f"✅ File {file_name} loaded successfully and is ready for queries!"}

def load_and_process_csv(file_path):
    """Loads CSV and processes it for search & SQL queries."""
    global df, index, current_file_name, current_columns, conn

    with lock:  # Ensures thread safety for shared resources
        try:
            df = pd.read_csv(file_path, dtype=str, low_memory=False).fillna("N/A")
            print(f"Loaded CSV: {file_path} with shape {df.shape}")  # Debugging statement
            current_columns = df.columns.tolist()  # Update current_columns
            current_file_name = os.path.basename(file_path)  # Update current_file_name
            print(f"Current File Name: {current_file_name}")  # Debugging statement
            print(f"Current Columns: {current_columns}")  # Debugging statement
            
            # Load Data into SQLite
            df.to_sql("data", conn, if_exists="replace", index=False)  # Create the table in SQLite
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

def chatbot_interface(uploaded_file, selected_file):
    """Handles file uploads and selection."""
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())
        load_and_process_csv(file_path)

    if selected_file:
        load_and_process_csv(os.path.join(UPLOAD_FOLDER, selected_file))

def update_ui(uploaded_file, selected_file):
    """Update UI when CSV is changed."""
    chatbot_interface(uploaded_file, selected_file)
    uploaded_files = get_uploaded_files()
    
    # Ensure current file name and columns are updated correctly
    current_file_name_display = current_file_name if current_file_name else "No CSV loaded"
    
    # CSV File Info: Show number of rows and columns
    csv_file_info = f"✅ CSV Loaded! {df.shape[0]} rows, {df.shape[1]} columns." if df is not None else "No CSV loaded."
    
    # CSV Data Column Names: Join column names with a comma
    current_columns_display = ", ".join(current_columns) if current_columns else "No columns available"

    return current_file_name_display, csv_file_info, current_columns_display

class QueryRequest(BaseModel):
    query: str

def run_pandas_query(query):
    """Handles Pandas query execution."""
    return asyncio.run(handle_pandas_query(QueryRequest(query=query)))

@app.post("/query/pandas/")
async def handle_pandas_query(request: QueryRequest):
    """Executes user queries in Pandas syntax and returns output in tabular format."""
    query = request.query  # Access the query from the request object
    if not query or df is None:
        raise HTTPException(status_code=400, detail="❌ No query entered or no CSV loaded.")
    
    print(f"Received Pandas query: {query}")  # Debugging statement
    try:
        result = eval(query, {"df": df})
        if isinstance(result, pd.DataFrame):
            return result.to_html(index=False, classes="table table-striped")  # Return HTML directly
        else:
            return str(result)  # Ensure this is a string
    except Exception as e:
        return f"❌ Error executing query: {str(e)}"  # Return error message as a string

class SQLQueryRequest(BaseModel):
    query: str

def run_sql_query(query):
    """Handles SQL query execution."""
    return asyncio.run(handle_sql_query(SQLQueryRequest(query=query)))

@app.post("/query/sql/")
async def handle_sql_query(request: SQLQueryRequest):
    """Executes SQL queries on the CSV data stored in SQLite."""
    query = request.query  # Access the query from the request object
    if not query or conn is None:
        raise HTTPException(status_code=400, detail="❌ No query entered or database not connected.")

    print(f"Received SQL query: {query}")  # Debugging statement
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df_result = pd.DataFrame(result, columns=column_names)

        return df_result.to_html(index=False, classes="table table-striped")  # Return HTML directly
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ SQL Error: {str(e)}")

@app.get("/columns/") 
def get_columns():
    """Returns the current CSV's column names."""
    if current_columns:
        return {"columns": current_columns}
    else:
        raise HTTPException(status_code=400, detail="No CSV loaded.")

class VisualizationRequest(BaseModel):
    x_column: str
    y_column: str
    chart_type: str

@app.post("/visualize/")
def generate_plot(request: VisualizationRequest):
    """Generates a plot for the selected columns and chart type."""
    if df is None:
        raise HTTPException(status_code=400, detail="❌ No CSV loaded.")

    x_column, y_column, chart_type = request.x_column, request.y_column, request.chart_type

    if x_column not in df.columns or y_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"❌ Invalid columns selected. Available: {df.columns.tolist()}")

    df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')

    if df[x_column].isnull().all() or df[y_column].isnull().all():
        raise HTTPException(status_code=400, detail="❌ One of the columns has no valid numeric data for plotting.")

    plt.figure(figsize=(8, 6))

    if chart_type == "Bar Chart":
        sns.barplot(data=df, x=x_column, y=y_column)
        plt.title(f"Bar Chart of {y_column} vs {x_column}")
    elif chart_type == "Pie Chart":
        df[y_column].value_counts().plot(kind="pie", autopct='%1.1f%%')
        plt.title(f"Pie Chart of {y_column}")
    elif chart_type == "Histogram":
        sns.histplot(df[y_column], kde=True)
        plt.title(f"Histogram of {y_column}")
    else:
        raise HTTPException(status_code=400, detail="❌ Unsupported chart type.")

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    plt.close()

    return StreamingResponse(img_buffer, media_type="image/png")

# Gradio UI
def launch_gradio():
    with gr.Blocks() as gradio_ui:
        gr.Markdown("# 📊 AI-Powered CSV Chatbot - SQL & Pandas Queries")
        gr.Markdown("Upload a CSV file and query it using **Pandas syntax** or **SQL queries**.")

        # File upload & selection
        with gr.Row():
            uploaded_file = gr.File(label="📤 Upload CSV File")
            selected_file = gr.Dropdown(label="📂 Select Previous File", choices=get_uploaded_files(), interactive=True)

        load_btn = gr.Button("📥 Load File")
        file_name_display = gr.Textbox(label="Current CSV File", interactive=False)
        csv_file_info_display = gr.Textbox(label="CSV File Info", interactive=False)
        column_names_display = gr.Textbox(label="CSV Data Column Names", interactive=False)

        load_btn.click(update_ui, inputs=[uploaded_file, selected_file], outputs=[file_name_display, csv_file_info_display, column_names_display])

        # Tabs
        with gr.Tabs():
            with gr.Tab("Pandas Query Section"):
                user_query = gr.Textbox(label="📝 Enter Pandas Query", placeholder="e.g., df.head()")
                query_btn = gr.Button("🔍 Run Query")
                pandas_query_output = gr.HTML(label="Query Result")

                query_btn.click(fn=run_pandas_query, inputs=[user_query], outputs=[pandas_query_output])

                # Predefined queries
                predefined_query = gr.Dropdown(label="📊 Predefined Pandas Queries", choices=[ 
                    "df.head()", "df.tail()", "df.shape", "df.info()", "df.describe()", "df.columns", "df.dtypes", "df.nunique()", "df.isnull().sum()"
                ])
                predefined_query_btn = gr.Button("📊 Run Predefined Query")
                predefined_query_btn.click(fn=run_pandas_query, inputs=[predefined_query], outputs=[pandas_query_output])

            with gr.Tab("SQL Query Section"):
                sql_query = gr.Textbox(label="📝 Enter SQL Query", placeholder="e.g., SELECT * FROM data LIMIT 5;")
                sql_query_btn = gr.Button("🔍 Run SQL Query")
                sql_query_output = gr.HTML(label="SQL Query Result")

                sql_query_btn.click(fn=run_sql_query, inputs=[sql_query], outputs=[sql_query_output])

                # Predefined SQL queries
                predefined_sql_query = gr.Dropdown(label="📊 Predefined SQL Queries", choices=[
                    "SELECT * FROM data LIMIT 5;", 
                    "SELECT COUNT(*) FROM data;", 
                    "SELECT DISTINCT column_name FROM data;", 
                    "SELECT column1, column2 FROM data WHERE column1 = 'value';"
                ])
                predefined_sql_query_btn = gr.Button("📊 Run Predefined SQL Query")
                predefined_sql_query_btn.click(fn=run_sql_query, inputs=[predefined_sql_query], outputs=[sql_query_output])

            with gr.Tab("Data Visualization Section"):
                gr.Markdown("### Enter Columns for Visualization")
                x_column_for_plot = gr.Textbox(label="Enter X Column Name", placeholder="e.g., Column 1", interactive=True)
                y_column_for_plot = gr.Textbox(label="Enter Y Column Name", placeholder="e.g., Column 2", interactive=True)

                chart_type = gr.Dropdown(label="Select Chart Type", choices=["Bar Chart", "Pie Chart", "Histogram"])
                generate_plot_btn = gr.Button("Generate Plot")
                plot_output = gr.Image(label="Generated Plot")

                generate_plot_btn.click(fn=generate_plot, inputs=[x_column_for_plot, y_column_for_plot, chart_type], outputs=[plot_output])

    gradio_ui.launch(server_name="0.0.0.0", server_port=7861, share=True)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=7860), daemon=True)
    fastapi_thread.start()

    # Start Gradio UI
    launch_gradio()
#UDAY TEST
#UDAY TEST