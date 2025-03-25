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
import os
from email import policy
from email.parser import BytesParser
import pdfplumber
import docx
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# @@AUTHOR:Atosh Veer@@ 

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
DEEPSEEK_ENABLED = bool(os.getenv("DEEPSEEK_API_KEY"))  # Keeping for backward compatibility

print(f"OpenRouter enabled: {OPENROUTER_ENABLED}")
print(f"DeepSeek enabled: {DEEPSEEK_ENABLED}")  # Debug output

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
    'Money Movement Inbound': ['Principal', 'Interest', 'Principal+Interest', 'Principal+Interest+Fee'],
    'Money Movement Outbound': ['Timebound', 'Foreign Currency'],
    'AU Transfer': ['Reallocation Fees', 'Amendment Fees', 'Reallocation Principal'],
    'Closing Notice': ['Cashless Roll', 'Decrease', 'Increase'],
    'Fee Payment': ['Ongoing Fee', 'Letter of Credit Fee']
}

# ============================
# 3. CORE FUNCTION DEFINITIONS
# ============================

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
                                        if img_data:  # Check if image data exists
                                            images.append(Image.frombytes('RGB', (img['width'], img['height']), img_data))
                                    except Exception as img_error:
                                        logging.warning(f"Image processing failed on page {page.page_number}: {str(img_error)}")
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
    """Extract text from supported file types with enhanced error handling and OCR fallback"""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""

        # First try regular extraction
        text = ""
        if file_path.endswith(".pdf"):
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
                if not text:  # Fallback to OCR if no text found
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
        
        return text if text else extract_text_with_ocr(file_path)  # Final OCR fallback
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_eml_with_attachments(file_path):
    """Extracts both email body and attachments from EML with better attachment handling"""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
            
            # Get main body text
            text_part = msg.get_body(preferencelist=('plain', 'html'))
            text = text_part.get_content().strip() if text_part else ""
            
            # Process attachments
            attachments_text = []
            for part in msg.iter_attachments():
                try:
                    if part.get_content_type() == 'text/plain':
                        content = part.get_content()
                        if content:
                            attachments_text.append(content.strip())
                    elif part.get_content_type() in ['application/pdf', 
                                                   'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                        # Save temporary attachment for processing
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_path = temp_file.name
                            temp_file.write(part.get_payload(decode=True))
                        
                        try:
                            attachments_text.append(extract_text_from_file(temp_path))
                        finally:
                            os.unlink(temp_path)
                except Exception as attachment_error:
                    logging.warning(f"Attachment processing failed: {str(attachment_error)}")
                    continue
            
            return text + "\n\n[ATTACHMENTS]\n" + "\n\n".join(attachments_text) if attachments_text else text
            
    except Exception as e:
        logging.error(f"Error processing EML attachments: {str(e)}")
        return extract_text_from_file(file_path)  # Fallback to original behavior
    
def extract_eml_metadata(file_path):
    """Extract metadata from EML files with better error handling"""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
            return {
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'subject': msg.get('subject', ''),
                'date': msg.get('date', ''),
                'cc': msg.get('cc', ''),
                'bcc': msg.get('bcc', '')
            }
    except Exception as e:
        logging.error(f"Error extracting EML metadata: {str(e)}")
        return {}

def validate_subtype(subtype, category):
    """Validate subtype against allowed values for category with case-insensitive check"""
    if category not in SUBTYPE_MAPPING:
        return None
    
    if not subtype:
        return None
        
    # Case-insensitive comparison
    subtype_lower = subtype.lower()
    for valid_subtype in SUBTYPE_MAPPING[category]:
        if valid_subtype.lower() == subtype_lower:
            return valid_subtype  # Return the properly cased version
    
    return None

def assign_to_team(classification, entities):
    """Assign the request to appropriate team based on classification and entities with enhanced logic"""
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
                # Handle different currency formats
                clean_amount = amount.replace('$','').replace(',','').replace(' ','')
                amounts.append(float(clean_amount))
            except ValueError:
                continue
        if amounts and max(amounts) > 50000:
            return 'Senior Operations Team'
    
    # Special case for international transfers
    if classification['label'] == 'Money Movement Outbound':
        if any(('foreign' in entity.lower() or 'international' in entity.lower()) 
               for entity in entities.get('GPE', [])):
            return 'International Transfers Team'
    
    return rules.get(classification['label'], 'General Servicing Team')

def query_openrouter(prompt, max_retries=3):
    """Robust OpenRouter API query with improved error handling and retry logic"""
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
        "model": "openai/gpt-3.5-turbo",  # Can be changed to other models
        "messages": [{
            "role": "system",
            "content": "You are a helpful financial email classification assistant."
        }, {
            "role": "user", 
            "content": prompt
        }],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"}  # Added to ensure JSON response
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
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
            elif response.status_code == 429:
                wait_time = 2 * (attempt + 1)
                logging.warning(f"Rate limited - waiting {wait_time} seconds")
                time.sleep(wait_time)
                continue
                
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
    """Robust DeepSeek API query with improved error handling and retry logic"""
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
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
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
            elif response.status_code == 429:
                wait_time = 2 * (attempt + 1)
                logging.warning(f"Rate limited - waiting {wait_time} seconds")
                time.sleep(wait_time)
                continue
                
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
    """Unified LLM classification with improved error handling and response validation"""
    system_prompt = """You are a financial email classification assistant. 
    Analyze the email content and return JSON with these exact fields:
    - category: One of the valid categories
    - subtype: Appropriate subtype or null
    - confidence: Confidence score (0-1)
    - reasoning: Explanation of classification
    - entities: Extracted financial entities
    
    Valid categories: """ + ', '.join(VALID_CATEGORIES)
    
    user_prompt = f"""Classify this financial email:
    
    RULES:
    1. category MUST be one of: {', '.join(VALID_CATEGORIES)}
    2. For interest rate changes, use "Adjustment"
    3. For money transfers, use appropriate Money Movement type
    4. Include all relevant entities
    
    Email Content:
    {text}
    
    Return valid JSON only with the required fields."""

    full_prompt = system_prompt + "\n\n" + user_prompt
    
    # Try OpenRouter first if enabled
    if OPENROUTER_ENABLED:
        response = query_openrouter(full_prompt)
        if response:
            try:
                content = response['choices'][0]['message']['content']
                result = json.loads(content)
                
                # Validate required fields
                if 'category' not in result or result['category'] not in VALID_CATEGORIES:
                    raise ValueError("Invalid or missing category in response")
                
                # Process confidence score
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp between 0.0 and 1.0
                    
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
                    logging.debug(f"Response content: {content[:200]}...")  # Log first 200 chars
    
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
                confidence = max(0.0, min(1.0, confidence))  # Clamp between 0.0 and 1.0
                    
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
        
        # Improved fallback logic with more categories
        label_map = {
            'LABEL_0': 'Money Movement Inbound',
            'LABEL_1': 'Adjustment',
            'LABEL_2': 'Fee Payment',
            'LABEL_3': 'Money Movement Outbound'
        }
        label = label_map.get(result['label'], 'Other')
        
        # Enhanced subtype detection
        subtype = None
        text_lower = text.lower()
        
        if label == 'Money Movement Inbound':
            if 'principal' in text_lower and 'interest' in text_lower and 'fee' in text_lower:
                subtype = 'Principal+Interest+Fee'
            elif 'principal' in text_lower and 'interest' in text_lower:
                subtype = 'Principal+Interest'
            elif 'principal' in text_lower:
                subtype = 'Principal'
            elif 'interest' in text_lower:
                subtype = 'Interest'
        elif label == 'Fee Payment':
            if 'letter of credit' in text_lower:
                subtype = 'Letter of Credit Fee'
            elif 'ongoing' in text_lower:
                subtype = 'Ongoing Fee'
        
        return {
            'label': label,
            'score': result['score'],
            'subtype': subtype,
            'reasoning': 'Classified by local model',
            'llm_entities': {},
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

def classify_email(text):
    """Main classification function with improved fallback hierarchy and validation"""
    if not text.strip():
        return {
            'label': 'Other',
            'score': 0.0,
            'subtype': None,
            'reasoning': 'Empty text',
            'llm_entities': {},
            'source': 'None'
        }
    
    # Try LLM classification first (OpenRouter -> DeepSeek)
    llm_result = classify_with_llm(text)
    if llm_result and llm_result['score'] >= 0.7:  # Only accept high-confidence LLM results
        return llm_result
    
    # Fallback to local model
    local_result = classify_with_fine_tuned_model(text)
    if local_result['score'] >= 0.6:  # Only accept decent-confidence local results
        return local_result
    
    # Final fallback
    return {
        'label': 'Other',
        'score': 0.0,
        'subtype': None,
        'reasoning': 'Low confidence in all classification methods',
        'llm_entities': {},
        'source': 'Fallback'
    }

def detect_duplicates(text, previous_emails, threshold=0.85):
    """Enhanced duplicate detection with better preview and performance"""
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
                        'preview': ' '.join(batch[j].split()[:15]) + ('...' if len(batch[j].split()) > 15 else '')
                    })
        
        return duplicates
    except Exception as e:
        logging.error(f"Duplicate detection failed: {str(e)}")
        return []

def extract_entities(text):
    """Comprehensive entity extraction with enhanced financial patterns"""
    try:
        doc = nlp(text)
        entities = {ent.label_: [] for ent in doc.ents}
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Enhanced financial patterns
        amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
        if amounts:
            entities["MONEY"] = list(set(amounts))  # Remove duplicates
            
        # Improved date detection
        date_patterns = [
            r'\b\d{1,2}[A-Z]{3}\d{2,4}\b',  # 01JAN2023
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b[A-Z][a-z]{2,8} \d{1,2},? \d{4}\b'  # January 1, 2023
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.I))
        if dates:
            entities["DATE"] = list(set(dates))
            
        # Enhanced account number detection
        account_numbers = re.findall(r'(?:Account|Acct)(?: #|:)?\s*([A-Z0-9-]{5,})', text, re.I)
        if account_numbers:
            entities["ACCOUNT"] = list(set(account_numbers))
            
        # Enhanced loan ID detection
        loan_ids = re.findall(r'Loan (?:ID|Number|No\.?)\s*[:#]?\s*([A-Z0-9-]{4,})', text, re.I)
        if loan_ids:
            entities["LOAN_ID"] = list(set(loan_ids))
            
        # Enhanced rate detection
        rates = re.findall(r'(?:Rate|Interest)\s*[:=]?\s*([\d.]+%?)', text, re.I)
        if rates:
            entities["INTEREST_RATE"] = list(set(rates))
            
        # Extract names
        names = re.findall(r'(?:Dear|Hello|Hi|To)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text, re.I)
        if names:
            entities["PERSON"] = list(set(names))
            
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

def test_api_connection():
    """Comprehensive API connection test with better reporting"""
    print("\nTesting API connections...")
    
    test_results = {
        'OpenRouter': False,
        'DeepSeek': False
    }
    
    # Test OpenRouter first
    if OPENROUTER_ENABLED:
        print("Testing OpenRouter connection...")
        test_prompt = "Respond with this exact text: 'API_TEST_OK'"
        response = query_openrouter(test_prompt)
        
        if response:
            try:
                content = response['choices'][0]['message']['content']
                if "API_TEST_OK" in content:
                    print("✅ OpenRouter API connection successful!")
                    test_results['OpenRouter'] = True
                else:
                    print(f"❌ Unexpected OpenRouter response: {content[:100]}...")
            except KeyError:
                print("❌ Invalid OpenRouter response format")
    
    # Test DeepSeek
    if DEEPSEEK_ENABLED:
        print("Testing DeepSeek connection...")
        test_prompt = "Respond with this exact text: 'API_TEST_OK'"
        response = query_deepseek(test_prompt)
        
        if response:
            try:
                content = response['choices'][0]['message']['content']
                if "API_TEST_OK" in content:
                    print("✅ DeepSeek API connection successful!")
                    test_results['DeepSeek'] = True
                else:
                    print(f"❌ Unexpected DeepSeek response: {content[:100]}...")
            except KeyError:
                print("❌ Invalid DeepSeek response format")
    
    if not any(test_results.values()):
        print("❌ No working API connections found")
    elif not all(test_results.values()):
        print("⚠️ Partial API connectivity - some services unavailable")
    
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
