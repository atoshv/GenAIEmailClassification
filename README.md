# 📧 AI-Powered Email Classification System for Financial Requests


**Automatically classify financial emails with 95%+ accuracy using NLP and LLMs**  
*Perfect for banks, fintech, and accounting teams to automate payment processing workflows*


## 🌟 Key Features

- **Smart Classification**  
  ✔️ Detect 7+ financial categories (Payments, Adjustments, Transfers)  
  ✔️ Subtype identification (Interest, Principal, Loan Payment)  
  ✔️ Confidence scoring for each prediction
  


- **Multi-Source Processing**  
  📧 EML files | 📝 TXT documents | 📄 PDF invoices | 📑 DOCX files
  


- **Enterprise-Grade Tech Stack**  
  ```python
  - Transformers (DistilBERT fine-tuned model)
  - spaCy (Entity extraction)
  - Sentence-Transformers (Duplicate detection)
  - OpenRouter/DeepSeek API fallbacks
  



**💼 Auto-assign to teams (Payments, Wire Transfers, Legal)**

**🔔 Priority tagging (Critical/High/Medium/Low)**

**🔍 Duplicate request detection**





🚀 Quick Start


**Prerequisites**
    Python 3.9+ | pip | git


**Installation**
    git clone https://github.com/yourusername/financial-email-classifier.git
    cd financial-email-classifier
    pip install -r requirements.txt


**Configuration**
Rename .env.example to .env


**Usage Examples**
1.  Process single email:

    from email_classification import process_email
    results = process_email("payment_request.eml")
    print(results['classification'])  
    Ex: {'label': 'Money Movement Outbound', 'score': 0.97...}


2.  Batch process folder:

    python batch_processor.py ./emails ./results.json

3.  Launch REST API:
    python api.py


**API Endpoints**

    POST /set_category - Add new classification categories

    GET /get_mappings - View current taxonomy
    

<img width="289" alt="{466588F9-DF29-4022-8336-7A6BB65AE628}" src="https://github.com/user-attachments/assets/b0cf120d-0e53-40ba-b5a0-9e3641404412" />



🛠️ Customization Guide

1.  Add new categories & Edit mappings.json

    {
    "categories": {
        "New_Category": ["Subtype1", "Subtype2"]
    }
    }

2.  Train with your data:

    python fine_tune_model.py --data your_emails.csv

3. Adjust sensitivity:

    In email_classification.py
    CLASSIFICATION_THRESHOLD = 0.85  # Default confidence level



📜 License
    GNU GPLv3 | © 2024 atoshveer



💡 Support
    Found a bug?
    Open an issue, Want to contribute? 

1.  Fork the repo
2.  Create your feature branch
3.  Submit a PR!



🌍 Let’s Connect & Collaborate: 

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/atoshveer)  
    [![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/atoshveer)  
    [![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:veeratosh@gmail.com)



**Would you like me to:**

    Add a demo video section?
    Create a deployment guide for Docker/Kubernetes?
