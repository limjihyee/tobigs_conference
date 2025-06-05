import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def collect_documents_with_disease_name(output_dir):
    documents, metadata = [], []
    for dirpath, _, filenames in os.walk(output_dir):
        if os.path.basename(dirpath) == "정의":
            for filename in filenames:
                if filename.endswith(".json"):
                    path = os.path.join(dirpath, filename)
                    try:
                        with open(path, encoding='utf-8') as f:
                            data = json.load(f)
                            disease = data.get("disease_name", {}).get("kor", "").strip()
                            question = data.get("question", "").strip()
                            answer = data.get("answer", "").strip()
                            if disease and question and answer:
                                doc = f"질병명: {disease}\n질문: {question}\n답변: {answer}"
                                documents.append(doc)
                                metadata.append({
                                    "path": path,
                                    "disease": disease,
                                    "category": data.get("disease_category", "알 수 없음")
                                })
                    except Exception as e:
                        print(f"Error at {path}: {e}")
    return documents, metadata

# 실행
docs, meta = collect_documents_with_disease_name("./definition_output")

print(f"문서 수집 완료: {len(docs)}개")

# KoE5 임베딩
model = SentenceTransformer("nlpai-lab/KoE5")
embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)

# FAISS 인덱스 생성 및 저장
dim = 1024
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "definition_faiss_index.faiss")

# 문서와 메타데이터 저장
with open("definition_documents.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("definition_metadata.pkl", "wb") as f:
    pickle.dump(meta, f)

print("벡터 DB 저장 완료: definition_faiss_index.faiss")
print("문서 및 메타데이터 저장 완료: definition_documents.pkl, definition_metadata.pkl")
