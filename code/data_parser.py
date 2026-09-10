import json
import re
from bs4 import BeautifulSoup

def parse_local_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    chunks = []
    
    # 1. 取得左側固定的「規格標題」(例如：作業系統、中央處理器)
    title_tags = soup.select('.multiple-spec-list .spec-column .multiple-title')
    titles = [t.get_text(strip=True) for t in title_tags]

    # 2. 取得上方的「型號名稱」
    model_tags = soup.select('.multiple-spec-list .sticky-header-wrapper .product-title')
    models = [t.get_text(strip=True) for t in model_tags]

    # 3. 取得右側規格區塊，每一個 .swiper-slide 就代表「一個型號的直列資料」
    slides = soup.select('.multiple-spec-list .content-column .swiper-wrapper .swiper-slide')

    if not titles or not slides or not models:
        print("⚠️ 找不到完整的比較表結構，請確認 HTML 檔案！")
        return []

    # 4. 雙層迴圈：將每個 slide (型號) 的資料與標題像拉鍊一樣 (zip) 對齊
    for i, slide in enumerate(slides):
        if i >= len(models):
            break
            
        model_name = models[i]
        
        # 抓取這個型號直列底下的所有規格格子
        value_tags = slide.select('.spec-item-list')
        
        # 將標題與格子一對一配對
        for title, val_tag in zip(titles, value_tags):
            val_text = val_tag.get_text(separator=" ", strip=True)
            val_text = re.sub(r'\s+', ' ', val_text).strip() # 清理多餘空白
            
            if val_text:
                chunks.append(f"{model_name} 的{title}為：{val_text}。")
                
    return chunks

if __name__ == "__main__":
    file_name = "AORUS MASTER 16 AM6H 筆記型電腦 產品規格 - GIGABYTE 技嘉科技.html"
    chunks = parse_local_html(file_name)
    
    print(f"✅ 成功萃取出 {len(chunks)} 個完美對齊的型號規格 Chunks：\n")
    for i, chunk in enumerate(chunks[:5]): # 只印前 5 個檢查
        print(f"[{i+1}] {chunk}")
    print("...\n")
        
    with open('specs_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print("💾 已儲存至 specs_chunks.json")