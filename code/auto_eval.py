import json
import os
import time
from collections import defaultdict
from llama_cpp import Llama
from vector_search import SimpleVectorDB 
from sentence_transformers import SentenceTransformer, util
import re

def load_data(db):
    try:
        with open('specs_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        db.add_texts(chunks)
    except FileNotFoundError:
        print("❌ 找不到 specs_chunks.json，請先執行 data_parser.py")
        exit()

# 終極脫水清洗函數：拔除商標、空白、中線，並轉小寫
def super_clean(text):
    return text.replace("™", "").replace("®", "").replace("©", "").replace(" ", "").replace("-", "").lower()

def main():
    os.makedirs("../report", exist_ok=True)
    report_file = "../report/evaluation_report.txt"
    error_file = "../report/error_cases.txt" # 新增：專門存錯誤案例
    
    print("⏳ 載入測試資料 (test_cases.json)...")
    try:
        with open('test_cases.json', 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 test_cases.json")
        return

    db = SimpleVectorDB()
    load_data(db)
    
    print("⏳ 正在載入語意裁判模型 (multilingual-e5-small)...")
    embedder = SentenceTransformer('intfloat/multilingual-e5-small')
    
    print("⏳ 正在載入本地 LLM 模型 (Qwen2.5-3B)...")
    llm = Llama(
        model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf", 
        n_gpu_layers=-1, 
        n_ctx=2048,      
        verbose=False    
    )

    total_weighted_score = 0.0
    total_weight = 0.0
    recall_hits = 0
    results_log = []
    
    ttft_list = []
    tps_list = []
    type_stats = defaultdict(lambda: {"score": 0.0, "weight": 0.0})

    ABSTRACT_TERMS = {"相同", "不同", "一樣", "有別", "規格表中未提供此資訊", "This information is not provided", "identical", "可以"}

    print(f"\n🚀 開始執行自動化評測...\n")
    print("-" * 60)

    for idx, case in enumerate(test_cases, 1):
        query = case['query']
        weight = case.get('weight', 1.0)
        q_type = case.get('type', 'general')
        expected_facts = case.get('expected_facts', [])
        
        total_weight += weight
        type_stats[q_type]["weight"] += weight
        
        print(f"[{idx}/{len(test_cases)}] [{q_type.upper()}] 測試問題: {query}")
        
        # ==========================================
        # 階段 A：檢索與 Recall (套用終極脫水清洗)
        # ==========================================
        search_results = db.search(query, top_k=15)
        context = "\n".join([f"- {res['text']}" for res in search_results])
        
        cleaned_context = super_clean(context)
        context_missing_facts = []
        
        for fact in expected_facts:
            if fact in ABSTRACT_TERMS:
                continue
            if super_clean(fact) not in cleaned_context:
                context_missing_facts.append(fact)

        if not context_missing_facts:
            recall_hits += 1
            recall_status = "✅ 檢索成功"
        else:
            recall_status = f"❌ 檢索遺漏: {context_missing_facts}"
            
        print(f"   > Recall@15: {recall_status}")
        
        # ==========================================
        # 動態語言判斷與 Prompt 注入
        # ==========================================
        # 1. 判斷題目是否為純英文 (沒有中文字)
        is_english_query = not bool(re.search(r'[\u4e00-\u9fa5]', query))
        
        # 2. 針對語言給予極度明確、具體的單一指令
        if is_english_query:
            lang_enforcement = (
                "CRITICAL WARNING: The user asked in ENGLISH. "
                "Even though the [Context] is in Traditional Chinese, you MUST translate the facts and reply in 100% PURE ENGLISH. "
                "DO NOT output ANY Chinese characters. If missing, reply EXACTLY 'This information is not provided'."
            )
        else:
            lang_enforcement = (
                "請用繁體中文回答。若規格表中未提供資訊，請回答 '規格表中未提供此資訊'。"
            )

        assistant_sys_prompt = (
            "You are a professional GIGABYTE laptop customer service AI.\n"
            "Strict Rules:\n"
            "1. Base your answer STRICTLY on the [Context].\n"
            f"2. {lang_enforcement}\n"
            "3. When comparing models, explicitly list the specific specs for EACH model.\n"
            "4. If the user asks about 'this laptop' without specifying, list BZH, BYH, and BXH explicitly."
        )
        
        rag_prompt = f"<|im_start|>system\n{assistant_sys_prompt}\n\n[Context]:\n{context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        # ==========================================
        # 階段 B：生成回答
        # ==========================================
        start_time = time.time()
        response_stream = llm(rag_prompt, max_tokens=800, temperature=0.0, stop=["<|im_end|>", "<|im_start|>"], stream=True)
        
        answer = ""
        first_token_time = None
        token_count = 0
        
        for chunk in response_stream:
            if first_token_time is None:
                first_token_time = time.time()
            answer += chunk["choices"][0]["text"]
            token_count += 1
            
        ttft = first_token_time - start_time if first_token_time else 0
        total_gen_time = time.time() - first_token_time if first_token_time else 0
        tps = token_count / total_gen_time if total_gen_time > 0 else 0
        
        ttft_list.append(ttft)
        tps_list.append(tps)

        print(f"   > AI 回答: {answer[:40].replace(chr(10), ' ')}...") 
        # ==========================================
        # 階段 C：Hybrid 事實查核 (加入拒答題嚴格防禦)
        # ==========================================
        missing_facts = []
        semantic_hits = []
        
        # 終極語言防禦：檢查題目與答案的語言一致性
        query_has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', query))
        answer_has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', answer))
        
        if not query_has_chinese and answer_has_chinese:
            missing_facts.append("Language Mismatch (English query answered with Chinese)")
        elif any("this information is not provided" in str(f).lower() for f in expected_facts) and answer_has_chinese:
            missing_facts.append("Language Mismatch (Expected English rejection, got Chinese)")
        else:
            answer_emb = embedder.encode(answer, convert_to_tensor=True)
            cleaned_answer = super_clean(answer)
            
            # 定義拒答標準詞
            REJECTION_TERMS = {"規格表中未提供此資訊", "thisinformationisnotprovided"}
            
            for fact in expected_facts:
                clean_fact = super_clean(fact)
                
                # 🛡️ 針對「拒答題」的嚴格防禦：禁止語意放水與廢話
                if clean_fact in REJECTION_TERMS:
                    if clean_fact not in cleaned_answer:
                        missing_facts.append(f"未拒答 (未命中: {fact})")
                    continue # 拒答題判斷完畢，直接跳過語意比對！
                
                # 1. 極速字串比對 (一般事實)
                if clean_fact in cleaned_answer:
                    continue
                    
                # 2. 語意相似度檢查 (一般事實)
                fact_emb = embedder.encode(fact, convert_to_tensor=True)
                cosine_score = util.cos_sim(answer_emb, fact_emb).item()
                
                if cosine_score >= 0.82:
                    semantic_hits.append(f"{fact}(sim:{cosine_score:.2f})")
                else:
                    missing_facts.append(fact)
        
        if not missing_facts:
            score = 1.0
            reason_msg = f"✅ 驗證通過" + (f" (語意命中: {semantic_hits})" if semantic_hits else "")
        else:
            score = 0.0
            reason_msg = f"❌ 驗證失敗 (遺漏關鍵字: {missing_facts})"

        weighted_score = score * weight
        total_weighted_score += weighted_score
        type_stats[q_type]["score"] += weighted_score
        
        print(f"   > 裁判理由: {reason_msg} -> Score: {int(score)}")
        print("-" * 60)
        
        results_log.append({
            "id": case.get("id", idx),
            "type": q_type,
            "query": query,
            "answer": answer,
            "recall": recall_status,
            "reasoning": reason_msg,
            "score": int(score),
            "weight": weight,
            "ttft": round(ttft, 2),
            "tps": round(tps, 1)
        })

    # ==========================================
    # 寫入報告與錯誤案例
    # ==========================================
    final_percentage = (total_weighted_score / total_weight) * 100 if total_weight > 0 else 0
    recall_percentage = (recall_hits / len(test_cases)) * 100
    
    # 寫入主報告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n🏆 GIGABYTE RAG 系統評測報告\n" + "="*60 + "\n")
        f.write(f"🌟 最終準確率: {final_percentage:.2f}% | 🎯 Recall@15: {recall_percentage:.2f}%\n\n")
        
        for log in results_log:
            f.write(f"[{log['type'].upper()}] Q: {log['query']}\n")
            f.write(f"A: {log['answer']}\n")
            f.write(f"Judge: {log['reasoning']} -> Score: {log['score']}\n")
            f.write("-" * 50 + "\n")

    # 獨立挑出錯誤案例 (Score == 0 或 檢索失敗)
    error_cases = [log for log in results_log if log['score'] == 0 or "❌" in log['recall']]
    
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n🚨 錯誤與待確認案例 (Error Cases)\n" + "="*60 + "\n")
        f.write(f"共發現 {len(error_cases)} 筆異常\n\n")
        for log in error_cases:
            f.write(f"[{log['type'].upper()}] ID: {log['id']} | Q: {log['query']}\n")
            f.write(f"A: {log['answer']}\n")
            f.write(f"Retrieval: {log['recall']}\n")
            f.write(f"Judge: {log['reasoning']}\n")
            f.write("*" * 50 + "\n")

    print(f"\n✅ 評測完成！最終評分: {final_percentage:.2f}% | Recall@15: {recall_percentage:.2f}%")
    print(f"📄 完整報告: {report_file}")
    if error_cases:
        print(f"🚨 發現 {len(error_cases)} 筆異常，已統整至: {error_file}")

if __name__ == "__main__":
    main()