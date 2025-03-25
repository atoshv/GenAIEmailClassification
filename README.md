📧 Gen AI Email Classification System 🤖
Automate financial email processing with AI-powered classification, OCR, and smart routing

✨ Key Features
✅ AI-Powered Classification

Automatically categorizes emails into:

Money Movement 💸

Adjustment 📉

Fee Payment 💰

And more!

✅ Multi-Format Support
📄 PDF | 📧 EML | 📝 TXT | 📑 DOC/DOCX

✅ Smart Document Processing

OCR for image-based PDFs (via Tesseract) 🔍

Priority Assignment (High/Medium/Low) ⚡

Duplicate Detection to avoid redundant processing ♻

✅ Seamless Integration

DeepSeek API support (optional) for advanced classification 🧠

Local fallback model ensures reliability even offline 💻

🛠 Installation

1. Prerequisites
Python 3.8+

Tesseract OCR (for image-based PDFs):

Windows:
choco install tesseract       # via Chocolatey
Mac/Linux:

brew install tesseract        # Mac
sudo apt install tesseract-ocr # Linux

2. Set Up Virtual Environment

python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate        # Mac/Linux

3. Install Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

🚀 Usage
Single File Processing

python email_classification.py   # Output: email_results.json
Batch Processing

python batch_processor.py input_folder output.json
Generate Sample Data (For Testing)

python create_sample_eml.py      # Creates sample_email.eml
python create_sample_pdf.py      # Creates financial_request.pdf
python create_sample_txt.py      # Creates sample_email.txt

⚙ Configuration
DeepSeek API (Optional for enhanced accuracy):
Create .env file: env

DEEPSEEK_API_KEY=your_api_key_here

Fine-Tuning (Optional):
Retrain the local model:


python fine_tune_model.py     # Output: ./fine-tuned-model

📋 Sample Request Types
Request Type	Sub-Types
Money Movement	Principal, Interest, Principal+Interest
Adjustment	Rate changes, Loan modifications
Fee Payment	Ongoing fees, Letter of Credit

📬 Support & Contact For questions or feedback:
📧 Email: veeratosh@gmail.com

Happy automating! 🎉
