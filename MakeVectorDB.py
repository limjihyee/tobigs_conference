import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# 1. 경로 설정
base_dir = "./answered_output/Training"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 2. 임베딩 모델 로드
embed_model = SentenceTransformer("nlpai-lab/KoE5")
embedding_dim = 1024  # KoE5 출력 차원
index = faiss.IndexFlatIP(embedding_dim)

documents = []
metadata = []

# 3. 모든 JSON 파일 탐색
def get_json_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".json"):
                yield os.path.join(dirpath, f)

# 4. 배치 임베딩 및 저장
batch_size = 100
batch_texts = []
batch_meta = []

for i, file_path in enumerate(get_json_files(base_dir)):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        question = data.get("question", "")
        answer = data.get("answer", "")
        full_text = f"질문: {question}\n답변: {answer}"
        batch_texts.append(full_text)

        meta = {
            "file_path": file_path,
            "disease_category": data.get("disease_category"),
            "disease_name": data.get("disease_name", {}).get("kor"),
            "intention": data.get("intention"),
        }
        batch_meta.append(meta)

    # 배치로 처리
    if len(batch_texts) >= batch_size:
        emb = embed_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        index.add(emb)
        documents.extend(batch_texts)
        metadata.extend(batch_meta)
        batch_texts, batch_meta = [], []

# 마지막 남은 배치
if batch_texts:
    emb = embed_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    index.add(emb)
    documents.extend(batch_texts)
    metadata.extend(batch_meta)

# 5. FAISS 및 메타데이터 저장
faiss.write_index(index, os.path.join(output_dir, "disease_index.faiss"))

with open(os.path.join(output_dir, "disease_documents.pkl"), "wb") as f:
    pickle.dump(documents, f)

with open(os.path.join(output_dir, "disease_metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)
