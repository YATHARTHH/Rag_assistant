import os
import json
import requests
import time
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

API_URL = "http://127.0.0.1:8000"

def get_jwt_token():
    # Attempt to signup and login a test runner user
    username = "test_runner"
    password = "SecurePassword123" # satisfies strength rules
    
    # Sign up
    requests.post(f"{API_URL}/auth/signup", json={"username": username, "password": password})
    
    # Login
    resp = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
    if resp.status_code == 200:
        return resp.json()["token"]
    raise RuntimeError("Failed to authenticate test runner.")

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    try:
        # Use sentence_bleu with smoothing to avoid 0.0 scores for short matches
        from nltk.translate.bleu_score import SmoothingFunction
        chencherry = SmoothingFunction()
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)
    except Exception:
        return 0.0

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    except Exception:
        return 0.0

def run_regression_suite():
    dataset_path = "./tests/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Golden dataset not found at {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    print("Connecting to backend server...")
    try:
        token = get_jwt_token()
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return
        
    print(f"Loaded {len(test_cases)} test cases. Starting evaluation...")
    print("=" * 80)
    
    headers = {"Authorization": f"Bearer {token}"}
    results = []
    
    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["expected_answer"]
        print(f"\nTest {idx}: {query}")
        
        payload = {
            "query": query,
            "username": "test_runner",
            "history": [],
            "model_name": "llama-3.3-70b-versatile",
            "temperature": 0.0,
            "top_k": 3,
            "vector_weight": 0.5,
            "window_size": 2,
            "enable_reranking": False,
            "rerank_pool": 10,
            "prompt_style": "Strict Fact-Only",
            "parent_retrieval": False
        }
        
        start_time = time.time()
        resp = requests.post(f"{API_URL}/chat", json=payload, headers=headers, stream=True)
        latency = time.time() - start_time
        
        full_text = ""
        eval_scores = {}
        
        if resp.status_code == 200:
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        token_text = line_str[6:]
                        if "__EVAL_START__" in token_text and "__EVAL_END__" in token_text:
                            try:
                                start_idx = token_text.find("__EVAL_START__") + 14
                                end_idx = token_text.find("__EVAL_END__")
                                json_data = token_text[start_idx:end_idx]
                                print(f"DEBUG: Found json_data: {json_data}")
                                eval_scores = json.loads(json_data)
                            except Exception as parse_err:
                                print(f"DEBUG: JSON parse error: {parse_err}")
                                pass
                        elif not token_text.startswith("__METADATA_START__"):
                            full_text += token_text
                            
            # Compute objective overlaps
            bleu = compute_bleu(expected, full_text)
            rouge = compute_rouge_l(expected, full_text)
            
            # Extract LLM scores
            faithfulness = eval_scores.get("faithfulness", 0.0)
            relevance = eval_scores.get("relevance", 0.0)
            precision = eval_scores.get("precision", 0.0)
            
            results.append({
                "query": query,
                "latency": latency,
                "bleu": bleu,
                "rouge": rouge,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "precision": precision
            })
            
            print(f"-> BLEU: {bleu:.2f} | ROUGE-L: {rouge:.2f}")
            print(f"-> Faithfulness: {faithfulness:.2f} | Relevance: {relevance:.2f} | Context Precision: {precision:.2f}")
        else:
            print(f"-> Request failed: {resp.status_code} - {resp.text}")
            
    print("\n" + "=" * 80)
    print("RAG QUALITY REGRESSION REPORT SUMMARY")
    print("=" * 80)
    print(f"{'Query':<50} | {'BLEU':<6} | {'ROUGE-L':<8} | {'Faithful':<8} | {'Relevance':<9}")
    print("-" * 90)
    for r in results:
        short_q = r["query"][:47] + "..." if len(r["query"]) > 50 else r["query"]
        print(f"{short_q:<50} | {r['bleu']:<6.2f} | {r['rouge']:<8.2f} | {r['faithfulness']:<8.2f} | {r['relevance']:<9.2f}")
        
if __name__ == "__main__":
    run_regression_suite()
