# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Pipeline V5.8 - Smart Retry (Databricks Version)
# MAGIC
# MAGIC **Features:**
# MAGIC - Smart retry for suspicious answers
# MAGIC - BGE reranker
# MAGIC - Few-shot prompting via Fireworks API
# MAGIC - Ensemble BM25 + QLD retrieval
# MAGIC
# MAGIC **Previous F1: 30.1% → Target: 32%+**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install Dependencies

# COMMAND ----------

# MAGIC %pip uninstall pyserini -y
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Install required packages

%pip install faiss-cpu sentence-transformers nltk openai -q
# 
# Uninstall current version and install one compatible with Java 17

# Restart Python to pick up new packages (Databricks specific)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Java Check (Databricks usually has Java pre-installed)

# COMMAND ----------

# MAGIC %pip install pyserini==0.21.0
# MAGIC import subprocess
# MAGIC result = subprocess.run(['java', '-version'], capture_output=True, text=True)
# MAGIC print(result.stderr)  # Java version prints to stderr

# COMMAND ----------

# %pip install pyserini==0.36.0 

# COMMAND ----------

import os

# First, find where libjvm.so actually is
!find /usr/lib/jvm -name "libjvm.so" 2>/dev/null

# COMMAND ----------

import os

# Set Java home and JVM path (try this first)
os.environ['JAVA_HOME'] = '/usr/lib/jvm/zulu17-ca-amd64'
os.environ['JVM_PATH'] = '/usr/lib/jvm/zulu17-ca-amd64/lib/server/libjvm.so'

# Now import pyserini
from pyserini.search import SimpleSearcher
searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
print("✅ Index loaded!")

# COMMAND ----------

# import os

# # Alternative paths to try
# os.environ['JAVA_HOME'] = '/usr/lib/jvm/zulu8-ca-amd64'
# os.environ['JVM_PATH'] = '/usr/lib/jvm/zulu8-ca-amd64/jre/lib/amd64/server/libjvm.so'

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Setup & Imports

# COMMAND ----------

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fireworks API - Your deployed model
FIREWORKS_MODEL = "YOUR_FIREWORKS_MODEL"

# Retrieval parameters (tuned)
BM25_K1 = 0.9
BM25_B = 0.2
QLD_MU = 2000
BM25_WEIGHT = 0.7
QLD_WEIGHT = 0.3

# Context configuration
CONTEXT_MAX_CHARS = 800
CONTEXT_MAX_CHARS_CURRENCY = 900
CONTEXT_TOP_N = 20
RETRIEVAL_K = 30
RERANK_TOP_K = 10
CROSS_ENCODER_MAX_CHARS = 1024

# LLM parameters
LLM_MAX_TOKENS = 30
LLM_TEMPERATURE = 0.3

# Smart retry configuration
RETRY_WITH_MORE_CONTEXT = True
RETRY_CONTEXT_TOP_N = 5
RETRY_CONTEXT_MAX_CHARS = 900

# Parallel processing (lower for Databricks shared clusters)
MAX_WORKERS = 10
llm_0=str(LLM_TEMPERATURE)[0]
llm_2=str(LLM_TEMPERATURE)[2]

# File paths - UPDATE THESE FOR YOUR DBFS PATHS
TRAIN_PATH = "rag_train.csv"
TEST_PATH = "rag_test.csv"
OUTPUT_PATH = f"submission_v58_temp_{llm_0}_{llm_2}_k{RETRIEVAL_K}_topk{RERANK_TOP_K}_topn{CONTEXT_TOP_N}.csv"
print(OUTPUT_PATH)
print("✅ Configuration loaded")

# COMMAND ----------

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print("✅ NLTK data downloaded")

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# # Verify Pyserini can use Java
# # from pyserini.search import SimpleSearcher
# # searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
# print("✅ Index loaded successfully!")

# COMMAND ----------

# from pyserini.search import SimpleSearcher
import pandas as pd
import json
import re
import string
from collections import Counter
from tqdm import tqdm
import openai

print("✅ Imports complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Fireworks API Setup

# COMMAND ----------

# Get API key from Databricks secrets
# First, set up your secret scope: 
#   databricks secrets create-scope --scope fireworks
#   databricks secrets put --scope fireworks --key api_key
try:
    fireworks_api_key = dbutils.secrets.get(scope="fireworks", key="api_key")
except:
    # Fallback: hardcode for testing (NOT recommended for production)
    fireworks_api_key = "YOUR_API_KEY"  # Replace with your key
    print("⚠️ Using hardcoded API key - set up Databricks secrets for production")

client = openai.OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=fireworks_api_key
)

# Test connection
try:
    test_response = client.chat.completions.create(
        model=FIREWORKS_MODEL,
        messages=[{"role": "user", "content": "Say 'API working'"}],
        max_tokens=10
    )
    print(f"✅ API test: {test_response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Index & Data

# COMMAND ----------

# # Load Pyserini index
# searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
# print("✅ Index loaded")

# COMMAND ----------

# Load data
df_train = pd.read_csv(TRAIN_PATH, converters={"answers": json.loads})
df_test = pd.read_csv(TEST_PATH, converters={"answers": json.loads})
print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load Cross-Encoder

# COMMAND ----------

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
print("✅ BGE Cross-encoder loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Entity Extraction & Query Variants

# COMMAND ----------

from nltk.corpus import stopwords

NLTK_STOPWORDS = set(stopwords.words('english'))

QUESTION_WORDS = {
    'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
    'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'would',
    'should', 'will', 'shall', 'may', 'might', 'must', 'have', 'has', 'had',
    'name', 'give', 'tell', 'find', 'list', 'describe', 'explain',
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'type', 'kind', 'sort', 'form', 'called', 'known', 'famous',
    'first', 'last', 'most', 'many', 'much', 'some', 'any', 'each',
    'state', 'country', 'city', 'place', 'person', 'thing', 'character'
}

ALL_STOPWORDS = NLTK_STOPWORDS | QUESTION_WORDS

def extract_entities_v5(question):
    """Extract meaningful entities from question."""
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        return quoted
    
    words = question.split()
    entities = []
    current_entity = []
    
    for word in words:
        clean = re.sub(r'[^\w\s]', '', word)
        if not clean:
            continue
        if clean[0].isupper() and clean.lower() not in ALL_STOPWORDS:
            current_entity.append(clean)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
    
    if current_entity:
        entities.append(' '.join(current_entity))
    
    if not entities:
        q_lower = question.lower()
        lowercase_patterns = [
            r"who (?:is|was|were|are) (.+?)(?:\?|$)",
            r"what (?:is|was|were|are) (.+?)(?:\?|$)",
            r"where (?:is|was|were|are) (.+?)(?:\?|$)",
        ]
        for pattern in lowercase_patterns:
            match = re.search(pattern, q_lower)
            if match:
                candidate = match.group(1).strip()
                candidate = re.sub(r'\b(the|a|an|called|known as|named)\b', '', candidate).strip()
                if candidate and candidate not in ALL_STOPWORDS:
                    entities.append(candidate)
                    break
    
    entities = [e for e in entities if len(e) > 2 or e.lower() not in ALL_STOPWORDS]
    return entities if entities else None

def generate_query_variants(question, entities):
    """Generate multiple query variants."""
    variants = []
    q_lower = question.lower()
    
    if entities:
        variants.append(' '.join(entities))
    
    clean_q = ' '.join(w for w in question.split() if w.lower() not in QUESTION_WORDS)
    if clean_q and clean_q not in variants:
        variants.append(clean_q)
    
    if entities:
        main_entity = entities[0]
        
        if any(w in q_lower for w in ['currency', 'money', 'dollar', 'peso']):
            variants.append(f"{main_entity} currency official money")
        elif any(w in q_lower for w in ['capital', 'capital city']):
            variants.append(f"{main_entity} capital city")
        elif any(w in q_lower for w in ['president', 'prime minister', 'leader']):
            variants.append(f"{main_entity} government leader head")
        elif any(w in q_lower for w in ['born', 'birth', 'birthplace']):
            variants.append(f"{main_entity} born birth birthplace")
        elif any(w in q_lower for w in ['play', 'played', 'character', 'role', 'actor']):
            variants.append(f"{main_entity} actor actress film movie role")
        elif any(w in q_lower for w in ['wrote', 'author', 'book', 'novel']):
            variants.append(f"{main_entity} author writer book novel")
        
        variants.append(f"{main_entity} wikipedia")
    
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    
    return unique[:5]

print("✅ Entity extraction ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Retrieval Functions

# COMMAND ----------

def retrieve_bm25(query, k=10):
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)
    hits = searcher.search(query, k)
    results = []
    for hit in hits:
        doc = searcher.doc(hit.docid)
        data = json.loads(doc.raw())
        results.append({
            'docid': hit.docid,
            'score': hit.score,
            'content': data['contents'].replace('\n', ' ')
        })
    return results

def retrieve_qld(query, k=10):
    searcher.set_qld(mu=QLD_MU)
    hits = searcher.search(query, k)
    results = []
    for hit in hits:
        doc = searcher.doc(hit.docid)
        data = json.loads(doc.raw())
        results.append({
            'docid': hit.docid,
            'score': hit.score,
            'content': data['contents'].replace('\n', ' ')
        })
    return results

def normalize_scores(results):
    if not results:
        return results
    scores = [r['score'] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r['norm_score'] = 1.0
    else:
        for r in results:
            r['norm_score'] = (r['score'] - min_s) / (max_s - min_s)
    return results

def ensemble_retrieve(query, k=10):
    bm25_results = retrieve_bm25(query, k=k*2)
    qld_results = retrieve_qld(query, k=k*2)
    
    bm25_results = normalize_scores(bm25_results)
    qld_results = normalize_scores(qld_results)
    
    bm25_scores = {r['docid']: r['norm_score'] for r in bm25_results}
    qld_scores = {r['docid']: r['norm_score'] for r in qld_results}
    
    all_docids = set(bm25_scores.keys()) | set(qld_scores.keys())
    
    content_map = {}
    for r in bm25_results + qld_results:
        if r['docid'] not in content_map:
            content_map[r['docid']] = r['content']
    
    fused = []
    for docid in all_docids:
        bm25_s = bm25_scores.get(docid, 0)
        qld_s = qld_scores.get(docid, 0)
        fused_score = BM25_WEIGHT * bm25_s + QLD_WEIGHT * qld_s
        fused.append({
            'docid': docid,
            'score': fused_score,
            'content': content_map[docid],
        })
    
    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused[:k]

def multi_query_ensemble_retrieve(question, k=10):
    entities = extract_entities_v5(question)
    variants = generate_query_variants(question, entities)
    
    if not variants:
        variants = [question]
    
    all_results = {}
    for variant in variants:
        results = ensemble_retrieve(variant, k=k)
        for r in results:
            docid = r['docid']
            if docid not in all_results:
                all_results[docid] = {
                    'docid': docid,
                    'content': r['content'],
                    'scores': [],
                    'variants_found': 0
                }
            all_results[docid]['scores'].append(r['score'])
            all_results[docid]['variants_found'] += 1
    
    final_results = []
    for docid, data in all_results.items():
        max_score = max(data['scores'])
        variant_bonus = 0.1 * (data['variants_found'] - 1)
        final_results.append({
            'docid': docid,
            'score': max_score + variant_bonus,
            'content': data['content']
        })
    
    final_results.sort(key=lambda x: x['score'], reverse=True)
    return final_results[:k]

def rerank_with_cross_encoder(question, results, top_k=None):
    if top_k is None:
        top_k = RERANK_TOP_K
    if not results:
        return []
    
    pairs = [(question, r['content'][:CROSS_ENCODER_MAX_CHARS]) for r in results]
    scores = cross_encoder.predict(pairs)
    
    for i, r in enumerate(results):
        r['ce_score'] = float(scores[i])
    
    results.sort(key=lambda x: x['ce_score'], reverse=True)
    return results[:top_k]

print("✅ Retrieval functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Few-Shot Prompt

# COMMAND ----------

def create_fewshot_prompt(query, contexts):
    """Few-shot prompt with examples."""
    context = '\n\n'.join(contexts)
    
    system_msg = """You are a precise question-answering assistant. Extract ONLY the specific answer from the context.

Rules:
- Think carefully step-by-step about the answer 
- Auto-correct the question if needed from the context
- Answer in 1-9 words maximum
- Give the specific name, place, thing, or value
- If you are asked to supply an answer that might have a list of possbile answer just spit them out one by the other
- Never EVER repeat the question nor answer in a conversational manner
- Do not include any word from the question in your answer
- Never EVER explain or elaborate, remain concise and to the point
- If unsure, give your best guess from the context"""

    example_1_user = """Context: The official currency of Jamaica is the Jamaican dollar, which has been in use since 1969. The currency code is JMD.

Question: What type of currency is used in Jamaica?"""
    example_1_assistant = "Jamaican dollar"

    example_2_user = """Context: Natalie Portman portrayed Padmé Amidala in the Star Wars prequel trilogy. She appeared in The Phantom Menace, Attack of the Clones, and Revenge of the Sith.

Question: Who played Padmé Amidala?"""
    example_2_assistant = "Natalie Portman"

    example_3_user = """Context: Michael Saul Dell is an American businessman and philanthropist. He is the founder and CEO of Dell Technologies, one of the world's largest technology infrastructure companies. Dell started the company in 1984.

Question: Who founded Dell computers?"""
    example_3_assistant = "Michael Dell"

    example_4_user = """Context: The West African CFA franc is the currency of eight independent states in West Africa: Benin, Burkina Faso, Guinea-Bissau, Ivory Coast, Mali, Niger, Senegal, and Togo.

Question: What currency does Senegal use?"""
    example_4_assistant = "West African CFA franc"

    actual_user = f"""Context: {context}

Question: {query}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example_1_user},
        {"role": "assistant", "content": example_1_assistant},
        {"role": "user", "content": example_2_user},
        {"role": "assistant", "content": example_2_assistant},
        {"role": "user", "content": example_3_user},
        {"role": "assistant", "content": example_3_assistant},
        {"role": "user", "content": example_4_user},
        {"role": "assistant", "content": example_4_assistant},
        {"role": "user", "content": actual_user},
    ]
    
    return messages

print("✅ Few-shot prompt ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. LLM Generation

# COMMAND ----------

def llm_generate(messages, max_tokens=None):
    """Generate answer via Fireworks API."""
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS
    
    try:
        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return "unknown"

print("✅ LLM generation ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Smart Retry System

# COMMAND ----------

SUSPICIOUS_ANSWERS = {
    "unknown", "i dont know", "i don't know", "dont know", "don't know","do not",
    "the", "a", "an", "it", "this", "that", "yes", "no", "none",
    "not found", "no answer", "cannot", "can't", "n/a", "na",
    "i'm not sure", "im not sure", "unclear", "not sure",
    "based on", "according to", "the context", "no information"
}

def is_suspicious(answer):
    """Check if answer looks suspicious and needs retry."""
    if not answer:
        return True
    
    clean = answer.lower().strip().rstrip('.,!?')
    
    if clean in SUSPICIOUS_ANSWERS:
        return True
    
    if len(clean) < 2:
        return True
    
    if clean.isdigit() and len(clean) == 1:
        return True
    
    hedges = ["i think", "i believe", "maybe", "perhaps", "possibly", 
              "it seems", "it appears", "i'm not", "im not", "cannot"]
    for hedge in hedges:
        if clean.startswith(hedge):
            return True
    
    if "context" in clean or "passage" in clean:
        return True
    
    return False

def get_context_with_params(question, top_n, max_chars):
    """Get context with specified parameters."""
    results = multi_query_ensemble_retrieve(question, k=RETRIEVAL_K)
    
    if not results:
        return []
    
    reranked = rerank_with_cross_encoder(question, results, top_k=top_n + 2)
    
    q_lower = question.lower()
    is_currency = any(w in q_lower for w in ['money', 'currency', 'dollar', 'peso', 'franc'])
    actual_max_chars = max(max_chars, CONTEXT_MAX_CHARS_CURRENCY) if is_currency else max_chars
    
    contexts = [r['content'][:actual_max_chars] for r in reranked[:top_n]]
    return contexts

# def create_retry_prompt(query, contexts):
#     """More explicit prompt for retry attempts."""
#     context = '\n\n'.join(contexts)
    
#     system_msg = """You are answering a trivia question. You MUST provide an answer - never say "unknown" or "I don't know".

# Rules:
# - Extract the SPECIFIC answer from the context
# - Answer in 1-9 words only
# - If the exact answer isn't clear, give your best guess based on the context
# - Just give the answer, no explanation"""

#     messages = [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer (1-5 words):"},
#     ]
    
#     return messages
def create_retry_prompt(query, contexts):
    """More explicit prompt for retry attempts."""
    context = '\n\n'.join(contexts)
    
    system_msg = """You are answering a trivia question. You MUST provide an answer - never say "unknown" or "I don't know".

Rules:
- Think step-by-step about the answer 
- Auto-correct the question if needed from the context
- Answer in 1-10 words maximum
- Give the specific name, place, thing, or value
- If you are asked to supply an answer that might have a list of possbile answer just spit them out one by the other
- Never EVER repeat the question nor answer in a conversational manner
- Do not include any word from the question in your answer
- Never EVER explain or elaborate remain concise and to the point
- If unsure, give your best guess from the context"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"},
    ]
    
    return messages

print("✅ Smart retry system ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Answer Cleaning

# COMMAND ----------

def clean_answer(answer):
    """Clean the LLM output."""
    if not answer:
        return "unknown"
    
    answer = answer.split('\n')[0].strip()
    
    prefixes = [
        'the answer is', 'answer:', 'a:', 'based on', 'according to',
        'uses the', 'is the', 'are the', 'is called', 'known as'
    ]
    
    answer_lower = answer.lower()
    for prefix in prefixes:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            answer_lower = answer.lower()
    
    answer = answer.strip('"\'').rstrip('.,;:!?')
    
    words = answer.split()
    if len(words) > 5:
        answer = ' '.join(words[:5])
    
    return answer.strip() if answer.strip() else "unknown"

def post_process_answer(question, answer):
    """Post-process answer based on question type."""
    q_lower = question.lower()
    answer_clean = answer.strip()
    
    if any(w in q_lower for w in ['when', 'what year', 'which year']):
        match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', answer_clean)
        if match:
            return match.group(0)
    
    if any(w in q_lower for w in ['currency', 'money']):
        match = re.search(r'(?:the\s+)?(\w+(?:\s+\w+)?\s+(?:dollar|franc|euro|peso|pound|yen|rupee|krona|won|yuan|ruble|rand))', 
                         answer_clean, re.I)
        if match:
            return match.group(1)
    
    if any(w in q_lower for w in ['how many', 'how much', 'what number']):
        match = re.search(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', answer_clean)
        if match:
            return match.group(0)
    
    return answer_clean

print("✅ Answer cleaning ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Main Pipeline V5.8

# COMMAND ----------

def answer_question_v58(question, verbose=False):
    """V5.8 Pipeline with Smart Retry."""
    # First attempt
    contexts = get_context_with_params(question, CONTEXT_TOP_N, CONTEXT_MAX_CHARS)
    
    if not contexts:
        return "unknown"
    
    messages = create_fewshot_prompt(question, contexts)
    raw_answer = llm_generate(messages)
    answer = clean_answer(raw_answer)
    
    if verbose:
        print(f"  First attempt: {answer}")
    
    # Check if retry needed
    if is_suspicious(answer) and RETRY_WITH_MORE_CONTEXT:
        if verbose:
            print(f"  ⚠️ Suspicious answer, retrying...")
        
        retry_contexts = get_context_with_params(
            question, 
            RETRY_CONTEXT_TOP_N, 
            RETRY_CONTEXT_MAX_CHARS
        )
        
        if retry_contexts:
            retry_messages = create_retry_prompt(question, retry_contexts)
            retry_answer = llm_generate(retry_messages, max_tokens=20)
            retry_clean = clean_answer(retry_answer)
            
            if verbose:
                print(f"  Retry result: {retry_clean}")
            
            if not is_suspicious(retry_clean):
                answer = retry_clean
            elif len(retry_clean) > len(answer) and retry_clean != "unknown":
                answer = retry_clean
    
    # Post-process
    answer = post_process_answer(question, answer)
    
    return answer

print("✅ V5.8 pipeline ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Quick Test

# COMMAND ----------

print("🧪 Testing V5.8 pipeline...\n")

test_questions = [
    # "What type of currency is used in Jamaica?",
    # "Who played Padmé Amidala?",
    # "What is the capital of France?",
    # "Who wrote Harry Potter?",
    "what country is next to russia?",
    "where do the san francisco giants play home games?",
    "where did france colonise?",
    "what county is stockton ca located in?",
    "where is wellsville missouri?",
    "where did pablo picasso die?",
    "what children's books did suzanne collins wrote?",
    "what town was abe lincoln born in?",
    "what does the japanese call their country?",
    "what country did vasco da gama come from?",
    "what are all of michael jackson albums?",
    "who played eowyn in the lord of the rings movies?",
    "what discovery did sir isaac newton make about force?",
    "what instrument does justin bieber?"
]

for q in test_questions:
    answer = answer_question_v58(q, verbose=True)
    print(f"Q: {q}")
    print(f"A: {answer}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Validation on Train Sample

# COMMAND ----------

def normalize_answer_eval(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in set(string.punctuation))
    return ' '.join(s.split())

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer_eval(prediction).split()
    gt_tokens = normalize_answer_eval(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

# COMMAND ----------

# Run validation on 100 samples
from concurrent.futures import ThreadPoolExecutor, as_completed

sample = df_train.sample(100, random_state=42)
results = []

def process_val_row(row_data):
    idx, row = row_data
    pred = answer_question_v58(row['question'])
    f1 = max(f1_score(pred, gt) for gt in row['answers'])
    return {'question': row['question'], 'pred': pred, 'gold': row['answers'], 'f1': f1}

row_data = [(idx, row) for idx, row in sample.iterrows()]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_val_row, rd) for rd in row_data]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
        results.append(future.result())

avg_f1 = sum(r['f1'] for r in results) / len(results)
print(f"\n{'='*50}")
print(f"📊 Validation F1: {avg_f1*100:.2f}%")
print(f"{'='*50}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Full Test Run

# COMMAND ----------

import time

def process_single_question(row_data):
    qid, question = row_data
    try:
        answer = answer_question_v58(question)
        return (qid, answer)
    except Exception as e:
        print(f"Error on {qid}: {e}")
        return (qid, "unknown")

def run_parallel(df, max_workers=10):
    predictions = {}
    row_data = [(row['id'], row['question']) for _, row in df.iterrows()]
    
    print(f"🚀 Processing {len(row_data)} questions with {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_question, rd): rd[0] for rd in row_data}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            qid, answer = future.result()
            predictions[qid] = answer
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f}s ({elapsed/len(row_data):.2f}s per question)")
    
    return predictions

# COMMAND ----------

print("🚀 Running V5.8 on full test set...")
predictions = run_parallel(df_test, max_workers=MAX_WORKERS)
print(f"\n✅ Generated {len(predictions)} predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Generate Submission

# COMMAND ----------

df_submission = pd.DataFrame(predictions.items(), columns=['id', 'prediction'])
df_submission['id'] = df_submission['id'].astype(int)
df_submission['prediction'] = df_submission['prediction'].apply(
    lambda x: json.dumps([x], ensure_ascii=False)
)

df_submission = df_submission.sort_values('id').reset_index(drop=True)

# Save to DBFS
df_submission.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Submission saved: {OUTPUT_PATH}")
print(f"📝 Rows: {len(df_submission)}")
display(df_submission.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Download Submission
# MAGIC
# MAGIC To download from DBFS to your local machine:

# COMMAND ----------

# # Copy to a downloadable location
# dbutils.fs.cp(OUTPUT_PATH.replace("/dbfs", "dbfs:"), "submission_v58.csv")
# print("File ready at /tmp/submission_v58.csv")
# print("Use 'Files' menu or dbutils.fs.cp to download")

# COMMAND ----------

print("\n" + "="*50)
print("🎉 V5.8 DATABRICKS RUN COMPLETE!")
print("="*50)
print(f"\n📁 Submission: {OUTPUT_PATH}")
print(f"🤖 Model: {FIREWORKS_MODEL}")
print(f"🔄 Smart Retry: Enabled")