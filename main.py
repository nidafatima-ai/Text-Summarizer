import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="T5 Text Summarizer")

# Get absolute path to the directory containing main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Setup templates directory
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load model and tokenizer from current directory
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR)
print("Model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"summary": None, "original_text": ""}
    )

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, content: str = Form(...)):
    if not content.strip():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"summary": "Please enter valid text.", "original_text": content}
        )

    # Tokenize input text
    inputs = tokenizer("summarize: " + content, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        min_length=30, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"summary": summary, "original_text": content}
    )