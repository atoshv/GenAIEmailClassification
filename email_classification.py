"""
Email Classification System for Financial Requests
Copyright (C) 2024 atoshveer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
__author__ = "Atosh Veer <veeratosh@gmail.com>"
__license__ = "GPL-3.0"

import os
import re
import json
import time
import logging
import glob
import requests
import tempfile
import fitz  # PyMuPDF for PDF reading
from email import message_from_bytes
from email import policy
from email.parser import BytesParser
import pdfplumber
import docx
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import mimetypes
from thefuzz import fuzz

# ========================
# 1. CONFIGURATION SETUP
# ========================
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="email_classification.log",
)

# Initialize API flags with more robust checking
OPENROUTER_ENABLED = bool(os.getenv("OPENROUTER_API_KEY"))
DEEPSEEK_ENABLED = bool(os.getenv("DEEPSEEK_API_KEY"))

print(f"OpenRouter enabled: {OPENROUTER_ENABLED}")
print(f"DeepSeek enabled: {DEEPSEEK_ENABLED}")

if OPENROUTER_ENABLED:
    logging.info("OpenRouter API key found and enabled")
elif DEEPSEEK_ENABLED:
    logging.info("DeepSeek API key found and enabled")
else:
    logging.info("No API keys found - using local models only")

# ========================
# 2. MODEL INITIALIZATION
# ========================
try:
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    exit(1)

# Constants
VALID_CATEGORIES = {
    'Adjustment', 'AU Transfer', 'Closing Notice', 'Commitment Change',
    'Fee Payment', 'Money Movement Inbound', 'Money Movement Outbound'
}

SUBTYPE_MAPPING = {
    'Money Movement Inbound': ['Principal', 'Interest', 'Principal+Interest', 
                             'Principal+Interest+Fee', 'Loan Payment', 'Loan Repayment'],
    'Money Movement Outbound': ['Timebound', 'Foreign Currency'],
    'AU Transfer': ['Reallocation Fees', 'Amendment Fees', 'Reallocation Principal'],
    'Closing Notice': ['Cashless Roll', 'Decrease', 'Increase'],
    'Fee Payment': ['Ongoing Fee', 'Letter of Credit Fee']
}

# Global connection flag
connection_ok = False

def clean_entity_text(text):
    """Remove trailing junk from extracted entities (newlines, hyphens, etc.)"""
    return re.sub(r'[\n\-:].*$', '', text).strip()

def is_financial_content(text):
    """Check if text contains financial indicators with enhanced pattern matching"""
    financial_keywords = [
        'amount', 'transfer', 'payment', 'loan', 'account',
        'interest', 'fee', 'principal', 'settlement', 'value date',
        'disbursement', 'repayment', 'transaction', 'balance', 'funds'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in financial_keywords) or \
           bool(re.search(r'(?:\$|USD|EUR|GBP)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text))

def extract_text_with_ocr(file_path):
    """Extract text from image-based documents using OCR with improved error handling"""
    try:
        if file_path.endswith(('.pdf', '.doc', '.docx')):
            try:
                import pytesseract
                from PIL import Image
                if file_path.endswith('.pdf'):
                    with pdfplumber.open(file_path) as pdf:
                        images = []
                        for page in pdf.pages:
                            if page.images:
                                for img in page.images:
                                    try:
                                        img_data = img['stream'].get_data()
                                        if img_data:
                                            images.append(Image.frombytes('RGB', (img['width'], img['height']), img_data))
                                    except Exception as img_error:
                                        logging.warning(f"Image processing failed on page {page.page_number}: {str(img_error)}")
                                        continue
                        if images:
                            return "\n".join(pytesseract.image_to_string(img) for img in images)
            except ImportError:
                logging.warning("OCR dependencies not installed. Falling back to regular extraction.")
            except Exception as e:
                logging.warning(f"OCR attempt failed: {str(e)}")
        return ""
    except Exception as e:
        logging.error(f"OCR extraction failed: {str(e)}")
        return ""

def extract_text_from_file(file_path):
    """Extract text from supported file types with enhanced error handling"""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""

        text = ""
        if file_path.endswith(".pdf"):
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
                if not text:
                    text = extract_text_with_ocr(file_path)
            except Exception as pdf_error:
                logging.error(f"PDF extraction error: {str(pdf_error)}")
                text = extract_text_with_ocr(file_path)
        elif file_path.endswith(".docx"):
            try:
                doc = docx.Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs if para.text).strip()
            except Exception as docx_error:
                logging.error(f"DOCX extraction error: {str(docx_error)}")
        elif file_path.endswith(".eml"):
            try:
                with open(file_path, "rb") as f:
                    msg = BytesParser(policy=policy.default).parse(f)
                    text_part = msg.get_body(preferencelist=('plain',))
                    if text_part:
                        text = text_part.get_content().strip()
            except Exception as eml_error:
                logging.error(f"EML extraction error: {str(eml_error)}")
        elif file_path.endswith(".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as txt_error:
                logging.error(f"TXT extraction error: {str(txt_error)}")
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return ""
        
        return text if text else extract_text_with_ocr(file_path)
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_eml_metadata(file_path):
    """Extract metadata from EML files with better error handling"""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
            metadata = {
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'subject': msg.get('subject', ''),
                'date': msg.get('date', ''),
                'cc': msg.get('cc', ''),
                'bcc': msg.get('bcc', ''),
                'message_id': msg.get('message-id', ''),
                'content_type': msg.get_content_type()
            }
            
            for header in ['x-priority', 'importance', 'in-reply-to']:
                if msg.get(header):
                    metadata[header] = msg.get(header)
            
            return metadata
    except Exception as e:
        logging.error(f"Error extracting EML metadata: {str(e)}")
        return {}

def validate_subtype(subtype, category):
    """Validate subtype with fuzzy matching"""
    if not subtype or category not in SUBTYPE_MAPPING:
        return None
        
    best_match, best_score = None, 0
    for valid_subtype in SUBTYPE_MAPPING[category]:
        score = fuzz.ratio(subtype.lower(), valid_subtype.lower())
        if score > best_score and score > 70:
            best_match, best_score = valid_subtype, score
            
    return best_match

def assign_to_team(classification, entities):
    """Assign the request to appropriate team with enhanced logic"""
    rules = {
        'Money Movement Inbound': 'Payment Processing Team',
        'Money Movement Outbound': 'Wire Transfer Team',
        'Adjustment': 'Loan Servicing Team',
        'AU Transfer': 'Account Management Team',
        'Closing Notice': 'Loan Operations Team',
        'Fee Payment': 'Billing Team',
        'Commitment Change': 'Underwriting Team'
    }
    
    # Special case for large amounts
    if 'MONEY' in entities:
        amounts = []
        for amount in entities['MONEY']:
            try:
                clean_amount = re.sub(r'[^\d.]', '', amount)
                amounts.append(float(clean_amount))
            except ValueError:
                continue
        if amounts and max(amounts) > 50000:
            return 'Senior Operations Team'
    
    # Special case for international transfers
    if classification['label'] == 'Money Movement Outbound':
        if any(('foreign' in entity.lower() or 'international' in entity.lower() or 
               'fx' in entity.lower() or 'currency' in entity.lower())
               for entity in entities.get('GPE', [])):
            return 'International Transfers Team'
    
    # Special case for legal documents
    if any(doc_type in entities.get('DOCUMENT_TYPE', [])
           for doc_type in ['contract', 'agreement', 'amendment']):
        return 'Legal Review Team'
    
    return rules.get(classification['label'], 'General Servicing Team')

def query_openrouter(prompt, max_retries=3):
    """Robust OpenRouter API query with improved error handling"""
    global OPENROUTER_ENABLED
    
    if not OPENROUTER_ENABLED:
        logging.info("OpenRouter disabled - skipping API call")
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("OpenRouter API key not configured")
        OPENROUTER_ENABLED = False
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Email Classification System"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{
            "role": "system",
            "content": "You are a helpful financial email classification assistant."
        }, {
            "role": "user", 
            "content": prompt
        }],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 2 * (attempt + 1)))
                logging.warning(f"Rate limited - waiting {retry_after} seconds")
                time.sleep(retry_after)
                continue
                
            if response.status_code == 400:
                logging.error(f"Bad request - check your prompt format. Response: {response.text}")
                continue
            elif response.status_code == 401:
                logging.error("Invalid API key - disabling OpenRouter")
                OPENROUTER_ENABLED = False
                return None
            elif response.status_code == 402:
                logging.error("Payment required - disabling OpenRouter")
                OPENROUTER_ENABLED = False
                return None
                
            response.raise_for_status()
            
            try:
                json_response = response.json()
                if not json_response.get('choices'):
                    logging.error("Invalid API response format - no choices")
                    continue
                return json_response
            except ValueError:
                logging.error("Invalid JSON response from API")
                continue
                
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(1 * (attempt + 1))
    
    logging.warning("Max retries reached for OpenRouter API")
    return None

def query_deepseek(prompt, max_retries=3):
    """Robust DeepSeek API query with improved error handling"""
    global DEEPSEEK_ENABLED
    
    if not DEEPSEEK_ENABLED:
        logging.info("DeepSeek disabled - skipping API call")
        return None

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DeepSeek API key not configured")
        DEEPSEEK_ENABLED = False
        return None

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "system",
            "content": "You are a helpful financial email classification assistant."
        }, {
            "role": "user", 
            "content": prompt
        }],
        "temperature": 0.1,
        "max_tokens": 300,
        "top_p": 0.9,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 2 * (attempt + 1)))
                logging.warning(f"Rate limited - waiting {retry_after} seconds")
                time.sleep(retry_after)
                continue
                
            if response.status_code == 400:
                logging.error(f"Bad request - check your prompt format. Response: {response.text}")
                continue
            elif response.status_code == 402:
                logging.error("Payment required - disabling DeepSeek")
                DEEPSEEK_ENABLED = False
                return None
            elif response.status_code == 401:
                logging.error("Invalid API key - disabling DeepSeek")
                DEEPSEEK_ENABLED = False
                return None
                
            response.raise_for_status()
            
            try:
                json_response = response.json()
                if not json_response.get('choices'):
                    logging.error("Invalid API response format - no choices")
                    continue
                return json_response
            except ValueError:
                logging.error("Invalid JSON response from API")
                continue
                
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(1 * (attempt + 1))
    
    logging.warning("Max retries reached for DeepSeek API")
    return None

def classify_with_llm(text):
    """Enhanced LLM classification with better money movement detection"""
    # Detect money movement direction
    text_lower = text.lower()
    direction = None
    if any(kw in text_lower for kw in ['received', 'deposit', 'credited', 'payment received']):
        direction = 'Inbound'
    elif any(kw in text_lower for kw in ['send', 'transfer', 'wire', 'pay to', 'disburse']):
        direction = 'Outbound'

    system_prompt = f"""You are a financial email classification assistant. 
    Analyze the email content and return JSON with these exact fields:
    - category: One of {', '.join(VALID_CATEGORIES)}
    - subtype: Appropriate subtype or null
    - confidence: Confidence score (0-1)
    - reasoning: Explanation of classification
    - entities: Extracted financial entities"""
    
    user_prompt = f"""Classify this financial email carefully:
        
    Key Indicators:
    - INBOUND if mentions: received, deposit, payment, interest
    - OUTBOUND if mentions: send, transfer, wire, pay

    Email Content:
    {text[:10000]}
    
    Return valid JSON only with the required fields."""

    full_prompt = system_prompt + "\n\n" + user_prompt
    
    # Try OpenRouter first if enabled
    if OPENROUTER_ENABLED:
        response = query_openrouter(full_prompt)
        if response:
            try:
                content = response['choices'][0]['message']['content']
                result = json.loads(content)
                
                if 'category' not in result or result['category'] not in VALID_CATEGORIES:
                    raise ValueError("Invalid or missing category in response")
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                    
                return {
                    'label': result['category'],
                    'subtype': validate_subtype(result.get('subtype'), result['category']),
                    'score': confidence,
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'llm_entities': result.get('entities', {}),
                    'source': 'OpenRouter'
                }
            except Exception as e:
                logging.error(f"OpenRouter response parsing failed: {str(e)}")
                if 'content' in locals():
                    logging.debug(f"Response content: {content[:200]}...")
    
    # Fallback to DeepSeek if enabled
    if DEEPSEEK_ENABLED:
        response = query_deepseek(full_prompt)
        if response:
            try:
                content = response['choices'][0]['message']['content']
                result = json.loads(content)
                
                if 'category' not in result or result['category'] not in VALID_CATEGORIES:
                    raise ValueError("Invalid or missing category in response")
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                    
                return {
                    'label': result['category'],
                    'subtype': validate_subtype(result.get('subtype'), result['category']),
                    'score': confidence,
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'llm_entities': result.get('entities', {}),
                    'source': 'DeepSeek'
                }
            except Exception as e:
                logging.error(f"DeepSeek response parsing failed: {str(e)}")
                if 'response' in locals():
                    logging.debug(f"Response content: {response['choices'][0]['message']['content'][:200]}...")
    
    return None

def classify_with_fine_tuned_model(text):
    """Reliable fallback classification with improved subtype detection"""
    try:
        classifier = pipeline(
            "text-classification",
            model="./fine-tuned-model",
            tokenizer="./fine-tuned-model"
        )
        result = classifier(text)[0]

        # Enhanced money movement direction detection
        text_lower = text.lower()
        if 'received' in text_lower or 'deposit' in text_lower or 'interest' in text_lower:
            direction = 'Inbound'
        elif 'send' in text_lower or 'transfer' in text_lower or 'pay' in text_lower:
            direction = 'Outbound'

        if 'interest' in text.lower() and 'payment' in text.lower():
            return {
                'label': 'Money Movement Inbound',
                'score': 0.95,
                'subtype': 'Interest',
                'reasoning': 'Interest payment received - inbound transaction',
                'llm_entities': extract_entities(text),
                'source': 'Local Model (Enhanced Logic)'
            }

        label_map = {
            'LABEL_0': 'Money Movement Inbound',
            'LABEL_1': 'Adjustment',
            'LABEL_2': 'Fee Payment',
            'LABEL_3': 'Money Movement Outbound'
        }

        label = label_map.get(result['label'], 'Other')
        
        subtype = None
        if hasattr(result, 'llm_entities') and result.get('llm_entities'):
            if 'Reference' in result['llm_entities']:
                subtype = validate_subtype(result['llm_entities']['Reference'], label)
            elif 'Type' in result['llm_entities']:
                subtype = validate_subtype(result['llm_entities']['Type'], label)

        if not subtype:
            text_lower = text.lower()
            if label == 'Money Movement Inbound':
                if 'loan payment' in text_lower:
                    subtype = 'Loan Payment'
                elif 'principal' in text_lower and 'interest' in text_lower:
                    subtype = 'Principal+Interest'
        
        return {
            'label': label,
            'score': result['score'],
            'subtype': subtype,
            'reasoning': 'Classified by local model',
            'llm_entities': result.get('llm_entities', {}),
            'source': 'Local Model'
        }
    except Exception as e:
        logging.error(f"Local model failed: {str(e)}")
        return {
            'label': 'Other',
            'score': 0.0,
            'subtype': None,
            'reasoning': 'Classification failed',
            'llm_entities': {},
            'source': 'Error'
        }

def inbound_classification(text):
    """Helper function for inbound payment classification"""
    return {
        'label': 'Money Movement Inbound',
        'score': 0.95,
        'subtype': 'Interest',
        'reasoning': 'Interest payment received - inbound transaction',
        'llm_entities': extract_entities(text),
        'source': 'Local Model (Enhanced Logic)'
    }

def outbound_classification(text):
    """Helper function for outbound payment classification"""
    return {
        'label': 'Money Movement Outbound',
        'score': 0.95,
        'subtype': None,
        'reasoning': 'Payment request detected - outbound transaction',
        'llm_entities': extract_entities(text),
        'source': 'Local Model (Enhanced Logic)'
    }

def classify_email(text):
    """Main classification function with improved money movement detection"""
    global connection_ok
    # More balanced direction detection
    text_lower = text.lower()
    
    # Strong inbound indicators
    strong_inbound = any(kw in text_lower for kw in [
        'received', 'deposit', 'remittance', 'credited', 'payment received'
    ])
    
    # Strong outbound indicators
    strong_outbound = any(kw in text_lower for kw in [
        'send', 'transfer', 'wire', 'disburse', 
        'pay to', 'pay account', 'debit', 'process payment'
    ])
    
    # Contextual indicators
    if 'interest' in text_lower and 'payment' in text_lower:
        return inbound_classification(text)
        
    if 'process this payment' in text_lower or 'please pay' in text_lower or 'request for money movement outbound' in text_lower:
        return outbound_classification(text)
    
    if not text.strip():
        return {
            'label': 'Other',
            'score': 0.0,
            'subtype': None,
            'reasoning': 'Empty text',
            'llm_entities': {},
            'source': 'None'
        }
    
    # Try LLM classification first
    llm_result = classify_with_llm(text)
    if llm_result and llm_result['score'] >= 0.7:
        if not connection_ok and 'OpenRouter' in llm_result['source']:
            llm_result['source'] = 'Local Model (Fallback)'
        return llm_result
    
    # Fallback to local model
    local_result = classify_with_fine_tuned_model(text)
    if local_result['score'] >= 0.6:
        return local_result
    
    # Final fallback with basic keyword matching
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ['transfer', 'send money', 'wire', 'pay to']):
        return {
            'label': 'Money Movement Outbound',
            'score': 0.5,
            'subtype': None,
            'reasoning': 'Detected money transfer keywords',
            'llm_entities': {},
            'source': 'Keyword Fallback'
        }
    elif any(keyword in text_lower for keyword in ['payment', 'received', 'deposit', 'credited']):
        return {
            'label': 'Money Movement Inbound',
            'score': 0.5,
            'subtype': None,
            'reasoning': 'Detected payment keywords',
            'llm_entities': {},
            'source': 'Keyword Fallback'
        }
    
    return {
        'label': 'Other',
        'score': 0.0,
        'subtype': None,
        'reasoning': 'Low confidence in all classification methods',
        'llm_entities': {},
        'source': 'Fallback'
    }

def detect_duplicates(text, previous_emails, threshold=0.85):
    """Enhanced duplicate detection with better preview, performance, and batch processing"""
    if not previous_emails or not text.strip():
        return []

    try:
        # Process in batches for large collections
        batch_size = 50
        duplicates = []
        
        for i in range(0, len(previous_emails), batch_size):
            batch = previous_emails[i:i+batch_size]
            
            current_embed = sentence_model.encode(text, convert_to_tensor=True)
            prev_embeds = sentence_model.encode(batch, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(current_embed, prev_embeds)[0]
            
            for j, sim in enumerate(similarities):
                if sim > threshold:
                    original_index = i + j
                    duplicates.append({
                        'index': original_index,
                        'score': float(sim),
                        'preview': ' '.join(batch[j].split()[:15]) + ('...' if len(batch[j].split()) > 15 else ''),
                        'similarity_percentage': f"{float(sim)*100:.1f}%"
                    })
        
        # Sort by similarity score (highest first)
        duplicates.sort(key=lambda x: x['score'], reverse=True)
        return duplicates[:5]  # Return top 5 matches
    except Exception as e:
        logging.error(f"Duplicate detection failed: {str(e)}")
        return []

def extract_entities(text):
    """Comprehensive entity extraction with enhanced patterns and cleaning"""
    try:
        def clean_entity(text):
            """Remove trailing junk but preserve important chars"""
            return re.sub(r'[\n\-].*$', '', text).strip()

        doc = nlp(text)
        entities = {ent.label_: [] for ent in doc.ents}
        
        # Enhanced account number pattern
        account_numbers = re.findall(
            r'(?:Account|Acct|Account No\.?)[:\s-]*([A-Za-z0-9-]{4,})', 
            text, 
            re.I
        )
        if account_numbers:
            entities["ACCOUNT"] = list(set(clean_entity(acc) for acc in account_numbers))
            
        # More robust date patterns
        date_patterns = [
            r'\b\d{1,2}[A-Z]{3}\d{2,4}\b',  # 25MAR25
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 25/03/2025
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ]

        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.I))
        if dates:
            entities["DATE"] = list(set(clean_entity(d) for d in dates))
            
        # Enhanced loan ID detection with cleaning
        loan_ids = re.findall(r'Loan (?:ID|Number|No\.?)\s*[:#]?\s*([A-Z0-9-]{4,})', text, re.I)
        if loan_ids:
            entities["LOAN_ID"] = list(set(clean_entity(lid) for lid in loan_ids))
            
        # Enhanced rate detection with cleaning
        rates = re.findall(r'(?:Rate|Interest)\s*[:=]?\s*([\d.]+%?)', text, re.I)
        if rates:
            entities["INTEREST_RATE"] = list(set(clean_entity(rate) for rate in rates))
            
        # Extract names with cleaning
        names = re.findall(
            r'(?:Dear|Hello|Hi|To|Attn:|Attention:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 
            text, 
            re.I
        )
        if names:
            entities["PERSON"] = list(set(clean_entity(name) for name in names))
            
        # Extract document references with cleaning
        documents = re.findall(
            r'(?:Document|Doc|Contract|Agreement)\s*(?:No\.?|Number)?\s*[:#]?\s*([A-Z0-9-]+)', 
            text, 
            re.I
        )
        if documents:
            entities["DOCUMENT_REF"] = list(set(clean_entity(doc) for doc in documents))
            
        # Extract email addresses (no cleaning needed)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            entities["EMAIL"] = list(set(emails))  # Emails don't need cleaning
            
        # Extract phone numbers with cleaning
        phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        if phones:
            entities["PHONE"] = list(set(clean_entity(ph) for ph in phones))
            
        return {k: v for k, v in entities.items() if v}
    except Exception as e:
        logging.error(f"Entity extraction failed: {str(e)}")
        return {}

def determine_priority(text, classification):
    """Enhanced priority detection with amount thresholds and keyword analysis"""
    text_lower = text.lower()
    
    # Immediate priority keywords
    urgent_keywords = ['urgent', 'asap', 'immediately', 'time sensitive', 'high priority']
    if any(word in text_lower for word in urgent_keywords):
        return 'critical'
    
    # Amount-based priority
    amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
    if amounts:
        try:
            max_amount = max(float(amt.replace('$','').replace(',','')) for amt in amounts)
            if max_amount >= 100000: return 'critical'
            if max_amount >= 10000: return 'high'
            if max_amount >= 5000: return 'medium'
        except ValueError:
            pass
    
    # Category-based defaults
    category_priority = {
        'Money Movement Inbound': 'high',
        'Money Movement Outbound': 'high',
        'Adjustment': 'medium',
        'Fee Payment': 'medium',
        'Closing Notice': 'medium',
        'Commitment Change': 'medium'
    }
    
    return category_priority.get(classification['label'], 'low')

def split_multi_requests(text):
    """Improved multi-request detection with better splitting logic"""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= 1:
        return [text]
    
    split_phrases = [
        'also', 'additionally', 'furthermore', 
        'second request', 'another request',
        'would also like', 'need to request',
        'please also', 'in addition'
    ]
    
    # Find split points based on phrases and sentence structure
    split_points = []
    for i, sentence in enumerate(sentences[1:], 1):  # Skip first sentence
        if (any(phrase in sentence.lower() for phrase in split_phrases) or
            re.match(r'^(second|next|another|additional)', sentence.lower())):
            split_points.append(i)
    
    if not split_points:
        return [text]
    
    parts = []
    start = 0
    for point in split_points:
        part = ' '.join(sentences[start:point])
        if part.strip():
            parts.append(part)
        start = point
    
    remaining = ' '.join(sentences[start:])
    if remaining.strip():
        parts.append(remaining)
    
    return parts if len(parts) > 1 else [text]

def process_email(file_path, previous_emails=None):
    """Complete email processing pipeline with enhanced error handling"""
    previous_emails = previous_emails or []
    
    try:
        text = extract_text_from_file(file_path)
        if not text.strip():
            logging.warning(f"Empty text extracted from {file_path}")
            return None
        
        # Skip non-financial files
        if not is_financial_content(text):
            logging.info(f"Skipping non-financial file: {file_path}")
            return None
            
        metadata = {}
        if file_path.endswith(".eml"):
            metadata = extract_eml_metadata(file_path)
        
        results = []
        for part in split_multi_requests(text):
            try:
                classification = classify_email(part)
                if not classification:
                    logging.warning(f"Classification failed for part of {file_path}")
                    continue
                    
                duplicates = detect_duplicates(part, previous_emails)
                spacy_entities = extract_entities(part)
                priority = determine_priority(part, classification)
                assigned_team = assign_to_team(classification, spacy_entities)
                
                # Combine entities from both spaCy and LLM
                all_entities = {**spacy_entities}
                if classification.get('llm_entities'):
                    all_entities['LLM_Entities'] = classification['llm_entities']
                
                results.append({
                    'file_path': file_path,
                    'metadata': metadata,
                    'text': part,
                    'classification': classification,
                    'duplicates': duplicates,
                    'entities': all_entities,
                    'priority': priority,
                    'assigned_team': assigned_team,
                    'processing_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as part_error:
                logging.error(f"Error processing part of {file_path}: {str(part_error)}")
                continue
        
        return results if results else None
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

# Constants
MAPPINGS_FILE = "mappings.json"

# Function to load the mappings
def load_mappings():
    """Load the category and sub-category mappings from the JSON file."""
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r") as f:
            return json.load(f)
    else:
        return {"categories": {}}

# Function to save the mappings
def save_mappings(mappings):
    """Save the updated mappings to the JSON file."""
    with open(MAPPINGS_FILE, "w") as f:
        json.dump(mappings, f, indent=4)

# Function to add or update categories and sub-categories
def set_category_subcategory(category, subcategories):
    """
    Add or update categories and subcategories.
    :param category: Main category name
    :param subcategories: List of sub-categories
    """
    mappings = load_mappings()

    if category in mappings['categories']:
        # Append new sub-categories without duplication
        existing_subcategories = set(mappings['categories'][category])
        existing_subcategories.update(subcategories)
        mappings['categories'][category] = list(existing_subcategories)
    else:
        # Add new category with sub-categories
        mappings['categories'][category] = subcategories

    save_mappings(mappings)
    return mappings

def test_api_connection():
    """Comprehensive API connection test with better diagnostics"""
    print("\nAPI Connection Diagnostics:")
    
    # Test network connectivity first
    try:
        print(f"Internet connectivity: {requests.get('https://google.com').status_code == 200}")
        print(f"OpenRouter reachable: {requests.get('https://openrouter.ai').status_code == 200}")
        print(f"DeepSeek reachable: {requests.get('https://api.deepseek.com').status_code == 200}")
    except Exception as e:
        print(f"⚠️ Network test failed: {str(e)}")

    test_results = {'OpenRouter': False, 'DeepSeek': False}
    
    if OPENROUTER_ENABLED:
        print("\nTesting OpenRouter...")
        try:
            response = query_openrouter("Respond with 'API_TEST_OK'")
            if response and "API_TEST_OK" in response.get('choices', [{}])[0].get('message', {}).get('content', ''):
                test_results['OpenRouter'] = True
                print("✅ OpenRouter working")
            else:
                print("❌ OpenRouter response invalid - check API key")
        except Exception as e:
            print(f"❌ OpenRouter test failed: {str(e)}")

    if DEEPSEEK_ENABLED:
        print("\nTesting DeepSeek...")
        try:
            response = query_deepseek("Respond with 'API_TEST_OK'")
            if response and "API_TEST_OK" in response.get('choices', [{}])[0].get('message', {}).get('content', ''):
                test_results['DeepSeek'] = True
                print("✅ DeepSeek working")
            else:
                print("❌ DeepSeek response invalid - check API key")
        except Exception as e:
            print(f"❌ DeepSeek test failed: {str(e)}")
    
    if not any(test_results.values()):
        print("\n❌ All API connections failed - using local models")
    return any(test_results.values())

if __name__ == "__main__":
    # Run connection test first
    connection_ok = test_api_connection()
    
    if not connection_ok:
        print("\n⚠️ Warning: Proceeding with local models only")
    
    logging.info("Starting enhanced email classification pipeline")
    
    # Test files - automatically filter to existing files
    test_files = [f for f in glob.glob("*.*") 
                if f.lower().endswith(('.pdf', '.eml', '.txt', '.doc', '.docx'))]
    
    if not test_files:
        logging.error("No test files found!")
        exit(1)
    
    previous_emails = [
        "Please transfer $5,000 immediately.",
        "Kindly adjust the interest rate on my loan.",
        "Requesting $15,000 for processing."
    ]
    
    all_results = []
    for file_path in test_files:
        print(f"\nProcessing: {file_path}")
        results = process_email(file_path, previous_emails)
        
        if results:
            all_results.extend(results)
            for result in results:
                print("\nClassification:", result['classification'])
                print("Priority:", result['priority'])
                print("Assigned Team:", result['assigned_team'])
                print("Duplicates:", [f"Score: {d['score']:.2f}" for d in result['duplicates']])
                print("Entities:", {k: v for k, v in result['entities'].items() if k != 'LLM_Entities'})
                print("=" * 60)
            
            previous_emails.extend([r['text'] for r in results])
    
    # Save comprehensive results
    if all_results:
        try:
            with open('email_results.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logging.info("Pipeline completed successfully. Results saved to email_results.json")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
    else:
        logging.warning("Pipeline completed but no results were generated")