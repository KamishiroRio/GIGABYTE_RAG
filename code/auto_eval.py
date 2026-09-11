import json
import os
import time
from llama_cpp import Llama
from vector_search import SimpleVectorDB 

def load_data(db):
    try:
        with open('specs_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        db.add_texts(chunks)
    except FileNotFoundError:
        print("❌ 找不到 specs_chunks.json，請先執行 data_parser.py")
        exit()

def main():
    # 1. 建立報告目錄
    os.makedirs("../report", exist_ok=True)
    report_file = "../report/evaluation_report.txt"
    
    # 2. 載入測資
    print("⏳ 載入測試資料 (test_cases.json)...")
    try:
        with open('test_cases.json', 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 test_cases.json，請先建立測資檔案。")
        return

    # 3. 初始化 RAG 系統 (完全對齊 rag_app.py 的配置)
    db = SimpleVectorDB()
    load_data(db)
    
    print("⏳ 正在載入 LLM 模型 (Qwen2.5-3B)...")
    llm = Llama(
        model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf", 
        n_gpu_layers=-1, 
        n_ctx=2048,      
        verbose=False    
    )

    total_weighted_score = 0.0
    total_weight = 0.0
    results_log = []

    print(f"\n🚀 開始執行自動化評測 (共 {len(test_cases)} 題)...\n")
    print("-" * 60)

    for idx, case in enumerate(test_cases, 1):
        query = case['query']
        weight = case.get('weight', 1.0) # 預設權重為 1.0
        total_weight += weight
        
        print(f"[{idx}/{len(test_cases)}] 測試問題: {query}")
        
        # ==========================================
        # 階段 A：RAG 助手生成回答 (完全對齊 rag_app.py)
        # ==========================================
        search_results = db.search(query, top_k=15)
        context = "\n".join([f"- {res['text']}" for res in search_results])
        
        assistant_sys_prompt = (
            "You are a professional GIGABYTE laptop customer service AI.\n"
            "Strict Rules:\n"
            "1. Base your answer STRICTLY on the [Context] provided below.\n"
            "2. If the [Context] does not contain the answer, reply EXACTLY with '規格表中未提供此資訊' or 'This information is not provided'. DO NOT guess, infer, or hallucinate.\n"
            "3. You MUST answer entirely in the EXACT SAME LANGUAGE as the user's question (Translate the context internally before answering. If asked in English, reply ONLY in English).\n"
            "4. When asked to compare, carefully identify the differences between models (e.g., BZH vs BXH) from the context.\n"
            "5. IMPORTANT: If the user asks about 'this laptop' without specifying the exact model (BZH, BYH, or BXH), you MUST explicitly list the specifications for ALL THREE models, or clearly state that the specification applies to the entire AM6H series."
        )
        
        rag_prompt = f"<|im_start|>system\n{assistant_sys_prompt}\n\n[Context]:\n{context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        # 這裡我們不需要 stream=True，直接讓它生成完畢
        response = llm(
            rag_prompt,
            max_tokens=300,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        answer = response['choices'][0]['text'].strip()

        # ==========================================
        # 階段 B：LLM-as-a-Judge 評分
        # ==========================================
        judge_sys_prompt = (
            "You are an impartial and strict evaluator for a RAG system. "
            "Evaluate if the [AI Answer] is highly relevant and factually correct based on the [Context].\n"
            "Rules:\n"
            "1. Score '1' if the answer correctly addresses the query using ONLY the context.\n"
            "2. Score '1' if the context lacks the info AND the AI correctly states it cannot find the info (e.g., '規格表中未提供此資訊').\n"
            "3. Score '0' if the AI hallucinates, gives wrong info, ignores the prompt rules, or answers based on outside knowledge.\n"
            "Output ONLY the number 1 or 0. No other text."
        )
        
        judge_prompt = (
            f"<|im_start|>system\n{judge_sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n[User Query]: {query}\n[Context]:\n{context}\n\n[AI Answer]: {answer}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        judge_res = llm(
            judge_prompt, 
            max_tokens=5, 
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        judge_output = judge_res['choices'][0]['text'].strip()
        
        # 容錯解析：只要模型輸出包含 '1' 就給分，否則為 0
        score = 1.0 if '1' in judge_output else 0.0
        weighted_score = score * weight
        total_weighted_score += weighted_score
        
        # 印出結果讓你在終端機也能監控
        print(f"   > AI 回答: {answer[:40].replace(chr(10), ' ')}...") 
        print(f"   > 裁判評分: {int(score)} (權重: {weight}) -> 獲得積分: {weighted_score}")
        print("-" * 60)
        
        results_log.append({
            "id": case.get("id", idx),
            "query": query,
            "answer": answer,
            "score": int(score),
            "weight": weight
        })

    # 4. 計算總分與寫入報告
    final_percentage = (total_weighted_score / total_weight) * 100 if total_weight > 0 else 0
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("🏆 GIGABYTE RAG 系統定性評測報告 (LLM-as-a-Judge)\n")
        f.write("="*60 + "\n")
        f.write(f"總測資筆數: {len(test_cases)}\n")
        f.write(f"總權重分母: {total_weight}\n")
        f.write(f"加權總積分: {total_weighted_score}\n")
        f.write(f"🌟 最終系統評分: {final_percentage:.2f}%\n")
        f.write("="*60 + "\n\n")
        f.write("📝 詳細測資紀錄:\n\n")
        for log in results_log:
            f.write(f"Q: {log['query']}\n")
            f.write(f"A: {log['answer']}\n")
            f.write(f"Score: {log['score']} (Weight: {log['weight']})\n")
            f.write("-" * 40 + "\n")

    print(f"\n✅ 評測完成！最終評分: {final_percentage:.2f}%")
    print(f"📄 評測報告已儲存至 {report_file}")

if __name__ == "__main__":
    main()