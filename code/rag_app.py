import time
import json
import subprocess
from llama_cpp import Llama
from vector_search import SimpleVectorDB 

def get_vram_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return int(result.strip())
    except Exception:
        return 0

def load_data(db):
    try:
        with open('specs_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        db.add_texts(chunks)
    except FileNotFoundError:
        print("❌ 找不到 specs_chunks.json，請確認資料解析是否成功。")
        exit()

def main():
    base_vram = get_vram_usage()
    print(f"🖥️  系統常駐 VRAM 用量: {base_vram} MB (Windows/背景程式佔用)")
    
    db = SimpleVectorDB()
    load_data(db)
    
    print("\n⏳ 正在載入 LLM 模型 (載入至 VRAM)...")
    llm = Llama(
        model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf", 
        n_gpu_layers=-1, 
        n_ctx=2048,      
        verbose=False    
    )
    
    peak_vram = get_vram_usage()
    app_vram = peak_vram - base_vram
    print(f"✅ LLM 模型載入完成！")
    print(f"   📊 AI 系統實際佔用 VRAM: {app_vram} MB")
    
    if app_vram <= 4096:
        print("   (🎉 完美符合 < 4096 MB 的硬體限制)\n")
    else:
        print("   (🚨 警告：AI 佔用超過 4096 MB)\n")
    
    while True:
        query = input("🧑‍💻 請輸入關於筆電規格的問題 (輸入 'q' 離開): ")
        if query.lower() == 'q':
            break
            
        print("🔍 正在檢索相關規格...")
        # 🔧 調整 1：將 Top-K 加大到 15，解決多意圖與比較題漏抓資料的問題
        search_results = db.search(query, top_k=15)
        context = "\n".join([f"- {res['text']}" for res in search_results])
        
        # 🔧 調整 2：針對比較與翻譯下達死命令
        system_prompt = (
            "You are a professional GIGABYTE laptop customer service AI.\n"
            "Strict Rules:\n"
            "1. Base your answer STRICTLY on the [Context] provided below.\n"
            "2. If the [Context] does not contain the answer, reply EXACTLY with '規格表中未提供此資訊' or 'This information is not provided'. DO NOT guess, infer, or hallucinate.\n"
            "3. You MUST answer entirely in the EXACT SAME LANGUAGE as the user's question (Translate the context internally before answering. If asked in English, reply ONLY in English).\n"
            "4. When asked to compare, carefully identify the differences between models (e.g., BZH vs BXH) from the context.\n"
            "5. IMPORTANT: If the user asks about 'this laptop' without specifying the exact model (BZH, BYH, or BXH), you MUST explicitly list the specifications for ALL THREE models, or clearly state that the specification applies to the entire AM6H series."
        )
        
        prompt = f"<|im_start|>system\n{system_prompt}\n\n[Context]:\n{context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        print("🤖 AI 助手回答：", end="", flush=True)
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        stream = llm(
            prompt,
            max_tokens=300,
            temperature=0.0, 
            stop=["<|im_end|>", "<|im_start|>"], 
            stream=True
        )
        
        for output in stream:
            if first_token_time is None:
                first_token_time = time.time()
                
            chunk_text = output['choices'][0]['text']
            print(chunk_text, end="", flush=True)
            token_count += 1
            
        end_time = time.time()
        print("\n")
        
        ttft = first_token_time - start_time
        total_gen_time = end_time - first_token_time
        tps = token_count / total_gen_time if total_gen_time > 0 else 0
        current_vram = get_vram_usage()
        current_app_vram = current_vram - base_vram
        
        print("-" * 50)
        print(f"📊 系統評測指標 Benchmark:")
        print(f"   - 首字延遲 (TTFT)  : {ttft:.3f} 秒")
        print(f"   - 生成速度 (TPS)   : {tps:.2f} tokens/秒")
        print(f"   - AI 實際 VRAM 佔用: {current_app_vram} MB (硬體限制 < 4096 MB)")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()