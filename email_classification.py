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
import requests
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
    """Extract text from image-based documents using OCR"""
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
                                    images.append(Image.frombytes('RGB', (img['width'], img['height']), img['stream'].get_data()))
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
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
            if not text:  # Fallback to OCR if no text found
                text = extract_text_with_ocr(file_path)
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text).strip()
        elif file_path.endswith(".eml"):
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
                text = msg.get_body(preferencelist=('plain',)).get_content().strip()
        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return ""
        
        return text if text else extract_text_with_ocr(file_path)  # Final OCR fallback
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_eml_metadata(file_path):
    """Extract metadata from EML files"""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
            return {
                'from': msg['from'],
                'to': msg['to'],
                'subject': msg['subject'],
                'date': msg['date']
            }
    except Exception as e:
        logging.error(f"Error extracting EML metadata: {str(e)}")
        return {}

def validate_subtype(subtype, category):
    """Validate subtype against allowed values for category"""
    if category not in SUBTYPE_MAPPING:
        return None
    return subtype if subtype in SUBTYPE_MAPPING[category] else None

def assign_to_team(classification, entities):
    """Assign the request to appropriate team based on classification and entities"""
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
                amounts.append(float(amount.replace('$','').replace(',','')))
            except ValueError:
                continue
        if amounts and max(amounts) > 50000:
            return 'Senior Operations Team'
    
    return rules.get(classification['label'], 'General Servicing Team')

def query_openrouter(prompt, max_retries=3):
    """Robust OpenRouter API query with proper error handling"""
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
        "max_tokens": 300
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
    """Robust DeepSeek API query with proper error handling"""
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
    """Unified LLM classification that tries OpenRouter first, then DeepSeek"""
    system_prompt = """You are a financial email classification assistant. 
    Analyze the email content and return JSON with these exact fields:
    - category: One of the valid categories
    - subtype: Appropriate subtype or null
    - confidence: Confidence score (0-1)
    - reasoning: Explanation of classification
    - entities: Extracted financial entities"""
    
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
                
                if 'category' not in result or result['category'] not in VALID_CATEGORIES:
                    raise ValueError("Invalid category")
                
                # Fixed the score calculation here
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
                logging.debug(f"Response content: {content if 'content' in locals() else 'N/A'}")
    
    # Fallback to DeepSeek if enabled
    if DEEPSEEK_ENABLED:
        response = query_deepseek(full_prompt)
        if response:
            try:
                result = json.loads(response['choices'][0]['message']['content'])
                
                if 'category' not in result or result['category'] not in VALID_CATEGORIES:
                    raise ValueError("Invalid category")
                
                # Fixed the score calculation here too
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
                logging.debug(f"Response content: {response['choices'][0]['message']['content']}")
    
    return None

def classify_with_fine_tuned_model(text):
    """Reliable fallback classification"""
    try:
        classifier = pipeline(
            "text-classification",
            model="./fine-tuned-model",
            tokenizer="./fine-tuned-model"
        )
        result = classifier(text)[0]
        
        # Improved fallback logic
        label = 'Money Movement Inbound' if result['label'] == 'LABEL_0' else 'Adjustment'
        
        # Infer subtype from text
        subtype = None
        if label == 'Money Movement Inbound':
            if 'principal' in text.lower() and 'interest' in text.lower():
                subtype = 'Principal+Interest'
            elif 'principal' in text.lower():
                subtype = 'Principal'
            elif 'interest' in text.lower():
                subtype = 'Interest'
        
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
    """Main classification function with proper fallback hierarchy"""
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
    result = classify_with_llm(text)
    if result:
        return result
    
    # Final fallback to local model
    return classify_with_fine_tuned_model(text)

def detect_duplicates(text, previous_emails, threshold=0.85):
    """Enhanced duplicate detection with better preview"""
    if not previous_emails:
        return []

    try:
        current_embed = sentence_model.encode(text, convert_to_tensor=True)
        prev_embeds = sentence_model.encode(previous_emails, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(current_embed, prev_embeds)[0]
        
        return [
            {
                'index': i,
                'score': float(sim),
                'preview': ' '.join(previous_emails[i].split()[:15]) + ('...' if len(previous_emails[i].split()) > 15 else '')
            }
            for i, sim in enumerate(similarities)
            if sim > threshold
        ]
    except Exception as e:
        logging.error(f"Duplicate detection failed: {str(e)}")
        return []

def extract_entities(text):
    """Comprehensive entity extraction with financial patterns"""
    try:
        doc = nlp(text)
        entities = {ent.label_: [] for ent in doc.ents}
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Enhanced financial patterns
        amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
        if amounts:
            entities["MONEY"] = amounts
            
        dates = re.findall(r'\b\d{1,2}[A-Z]{3}\d{2,4}\b', text)
        if dates:
            entities["DATE"] = dates
            
        account_numbers = re.findall(r'Account(?: #|:)?\s*([A-Z0-9-]+)', text, re.I)
        if account_numbers:
            entities["ACCOUNT"] = account_numbers
            
        loan_ids = re.findall(r'Loan (?:ID|Number)\s*:\s*([A-Z0-9-]+)', text, re.I)
        if loan_ids:
            entities["LOAN_ID"] = loan_ids
            
        rates = re.findall(r'(?:Rate|Interest)\s*:\s*([\d.]+%?)', text, re.I)
        if rates:
            entities["INTEREST_RATE"] = rates
            
        return {k: v for k, v in entities.items() if v}
    except Exception as e:
        logging.error(f"Entity extraction failed: {str(e)}")
        return {}

def determine_priority(text, classification):
    """Enhanced priority detection with amount thresholds"""
    text_lower = text.lower()
    
    # Immediate priority keywords
    if any(word in text_lower for word in ['urgent', 'asap', 'immediately']):
        return 'critical'
    
    amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
    if amounts:
        try:
            max_amount = max(float(amt.replace('$','').replace(',','')) for amt in amounts)
            if max_amount >= 10000: return 'high'
            if max_amount >= 5000: return 'medium'
        except ValueError:
            pass
    
    # Category-based defaults
    return {
        'Money Movement Inbound': 'high',
        'Money Movement Outbound': 'high',
        'Adjustment': 'medium',
        'Fee Payment': 'medium'
    }.get(classification['label'], 'low')

def split_multi_requests(text):
    """Improved multi-request detection"""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    split_phrases = ['also', 'additionally', 'furthermore', 'second request', 'another request']
    split_points = [i for i, s in enumerate(sentences) 
                   if any(phrase in s.lower() for phrase in split_phrases)]
    
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
    
    return parts

def process_email(file_path, previous_emails=None):
    """Complete email processing pipeline"""
    previous_emails = previous_emails or []
    text = extract_text_from_file(file_path)
    if not text:
        return None
    
    metadata = {}
    if file_path.endswith(".eml"):
        metadata = extract_eml_metadata(file_path)
    
    results = []
    for part in split_multi_requests(text):
        classification = classify_email(part)
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
    
    return results

def test_api_connection():
    """Comprehensive API connection test"""
    print("\nTesting API connections...")
    
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
                    return True
                else:
                    print(f"❌ Unexpected OpenRouter response: {content}")
            except KeyError:
                print("❌ Invalid OpenRouter response format")
    
    # Then test DeepSeek if OpenRouter failed
    if DEEPSEEK_ENABLED:
        print("Testing DeepSeek connection...")
        test_prompt = "Respond with this exact text: 'API_TEST_OK'"
        response = query_deepseek(test_prompt)
        
        if response:
            try:
                content = response['choices'][0]['message']['content']
                if "API_TEST_OK" in content:
                    print("✅ DeepSeek API connection successful!")
                    return True
                else:
                    print(f"❌ Unexpected DeepSeek response: {content}")
            except KeyError:
                print("❌ Invalid DeepSeek response format")
    
    print("❌ No working API connections found")
    return False

if __name__ == "__main__":
    # Run connection test first
    connection_ok = test_api_connection()
    
    if not connection_ok:
        print("\n⚠️ Warning: Proceeding with local models only")
    
    logging.info("Starting enhanced email classification pipeline")
    
    # Test files - automatically filter to existing files
    test_files = [f for f in [
        "sample_email.eml",
        "financial_request.pdf", 
        "sample_email.txt"
    ] if os.path.exists(f)]
    
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
                print("Entities:", result['entities'])
                print("=" * 60)
            
            previous_emails.extend([r['text'] for r in results])
    
    # Save comprehensive results
    with open('email_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logging.info("Pipeline completed successfully. Results saved to email_results.json")
