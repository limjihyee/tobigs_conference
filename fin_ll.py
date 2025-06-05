from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 사전 준비: KoE5 임베딩 모델 로드
embed_model = SentenceTransformer("nlpai-lab/KoE5-base")

# 2. FAISS 인덱스 및 메타데이터 로드
index = faiss.read_index("faiss_index.bin")
with open("documents.pkl", "rb") as f:
    data = pickle.load(f)
    documents = data["documents"]
    metadata = data["metadata"]

# 3. 사용자 질의 받기
query = "잇몸에서 피가 나는데 어떤 질병일 수 있어?"

# 4. 질의 임베딩
query_embedding = embed_model.encode([query])

# 5. FAISS 검색 (top-3)
k = 3
_, top_k_indices = index.search(query_embedding, k)

# 6. 검색된 문서 정리
retrieved_docs = [documents[i] for i in top_k_indices[0]]
context = "\n\n".join(retrieved_docs)

# LLM 프롬프트 구성 및 응답 생성 (3단계)
# 7. Qwen-4B LLM 로드
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", trust_remote_code=True, torch_dtype=torch.float16
).cuda()
model.eval()

# 8. 프롬프트 구성
prompt = f"""다음은 의료 관련 질병 정보야. 이 정보를 바탕으로 질문에 정확하고 간결하게 답해줘.

[배경 정보]
{context}

[질문]
{query}

[답변]
"""

# 9. LLM 질의
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n✅ 최종 답변:")
print(answer)