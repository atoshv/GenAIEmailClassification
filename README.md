# Gen AI Email Classification System
Gen AI-powered solution for automating email classification and OCR

## Features
- Automatic classification of financial emails
- Multi-format support (PDF, EML, TXT, DOC/DOCX)
- OCR for image-based documents
- Priority assignment and duplicate detection

## Installation
1. `pip install -r requirements.txt`
2. Install Tesseract OCR: `brew install tesseract` (Mac) or `sudo apt install tesseract-ocr` (Linux)

## Usage
Single file:
`python email_classification.py`

Batch processing:
`python batch_processor.py input_folder output.json`

## Configuration
Set DeepSeek API key in .env:
`DEEPSEEK_API_KEY=your_key_here`

Thanks. 
You can reach out for any doubts / help :-) 
Email: veeratosh@gmail.com
