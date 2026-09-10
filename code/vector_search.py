import json
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleVectorDB:
    def __init__(self, model_name="intfloat/multilingual-e5-small"):
        print(f"⏳ 正在載入 Embedding 模型: {model_name} (這會跑在 CPU 上以節省 VRAM)...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.chunks = []
        self.vectors = None
        
    def add_texts(self, texts):
        self.chunks = texts
        print(f"⏳ 開始將 {len(texts)} 筆 Chunk 轉換為向量...")
        
        # e5 模型的特殊要求：被檢索的文本前要加 "passage: "
        formatted_texts = [f"passage: {t}" for t in texts]
        
        # 轉換向量並直接做正規化 (L2 Normalize)
        self.vectors = self.model.encode(formatted_texts, normalize_embeddings=True)
        print("✅ 向量轉換完成！")
        
    def search(self, query, top_k=3):
        if self.vectors is None:
            raise ValueError("請先加入資料！")
            
        # e5 模型的特殊要求：搜尋的問句前要加 "query: "
        query_vector = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        
        # 【核心 RAG 演算法】：純 NumPy 計算內積。因為向量已正規化，內積即等於 Cosine Similarity
        similarities = np.dot(self.vectors, query_vector)
        
        # 排序並取出分數最高的前 K 個 Index
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "score": float(similarities[idx]),
                "text": self.chunks[idx]
            })
        return results

if __name__ == "__main__":
    try:
        with open('specs_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 specs_chunks.json，請先執行 data_parser.py")
        exit()
        
    db = SimpleVectorDB()
    db.add_texts(chunks)
    
    # 測試一下檢索準確度
    test_queries = [
        "請問這台筆電的顯卡是哪一張？",
        "記憶體最高可以擴充到多少？"
    ]
    
    for q in test_queries:
        print(f"\n🧑‍💻 使用者提問：{q}")
        results = db.search(q, top_k=2)
        for i, res in enumerate(results):
            print(f"  👉 Top {i+1} (相似度: {res['score']:.4f}): {res['text']}")