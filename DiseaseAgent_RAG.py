import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1) JSON 폴더 돌면서 문서 수집
def load_documents(data_dir="./definition_output"):
    docs = []
    for dirpath, _, files in os.walk(data_dir):
        for fn in files:
            if fn.endswith(".json"):
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as f:
                    j = json.load(f)
                d = j.get("disease_name",{}).get("kor","").strip()
                q = j.get("question","").strip()
                a = j.get("answer","").strip()
                if d and q and a:
                    docs.append(f"질병명: {d}\n질문: {q}\n답변: {a}")
    return docs

# 2) 매 실행 시 FAISS 인덱스 생성
def build_inmemory_index(docs, embed_model):
    embs = embed_model.encode(docs, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    dim  = embs.shape[1]
    idx  = faiss.IndexFlatIP(dim)
    idx.add(np.array(embs))
    return idx

# 3) 프롬프트 생성 함수
def make_prompt(query, retrieved_docs):
    ctx = "\n\n".join(f"[정의 {i+1}]\n{d}" for i,d in enumerate(retrieved_docs))
    return (
        f"사용자 증상: {query}\n\n"
        f"다음은 유사한 질병 정의입니다:\n{ctx}\n\n"
        "위 정의 중 가장 가능성 높은 질병명 하나만 골라 질병명과 판단 근거를 2줄 이하로 간단하게 답변하세요.\n"
        "질병명: \n"
        "판단 근거: \n"
    )

# 4) 
def main():
    # resource load
    docs        = load_documents("./definition_output")
    embed_model = SentenceTransformer("nlpai-lab/KoE5")
    idx         = build_inmemory_index(docs, embed_model)

    # LLM 파이프라인 설정
    model_id = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        device_map="auto", torch_dtype=torch.float16
    )
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=150, do_sample=False
    )

    print("=== 질병 RAG 서비스 시작 ===")
    while True:
        query = input("\n질문> ").strip()
        if not query:
            print("종료합니다.")
            break

        # 5) Retrieval
        qv = embed_model.encode([query], convert_to_numpy=True)
        _, I = idx.search(np.array(qv), k=3)
        retrieved = [docs[i] for i in I[0]]

        # 6) Generation
        prompt = make_prompt(query, retrieved)
        out    = generator(prompt)[0]["generated_text"][len(prompt):].strip()

        print("\n>>", out)

if __name__ == "__main__":
    main()