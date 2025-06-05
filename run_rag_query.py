import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 사용자 질문 예시
user_query = "기침이 2주 넘게 계속되고, 누런 가래가 나와요. 어떤 질병이 의심되나요?"

# 1. KoE5 임베딩 모델 로드 및 질의 임베딩
koe5 = SentenceTransformer("nlpai-lab/KoE5")
query_embedding = koe5.encode([user_query])

# 2. FAISS 인덱스 및 문서 로드
index = faiss.read_index("definition_faiss_index.faiss")
with open("definition_documents.pkl", "rb") as f: docs = pickle.load(f)
with open("definition_metadata.pkl", "rb") as f: meta = pickle.load(f)

# 3. Top-k 검색
k = 3
D, I = index.search(np.array(query_embedding), k)

# 4. 검색된 정의 문서들로 프롬프트 구성
prompt = f"""사용자 증상: {user_query}\n\n

다음은 유사한 질병 정의입니다:"""
for i in range(k):
    prompt += f"\n\n[정의 {i+1}]\n{docs[I[0][i]]}"

prompt += (
    "당신의 역할은 사용자 증상을 기반으로 위 정의 중 가장 가능성 높은 정의의 질병명 하나만 골라 답변하는 것이다.\n"
    "절대 두 개 이상 말하지 마. 아래 형식만 따라야해.\n"
    "질병명: 질병명을 한 줄로 간단히.\n"
    "판단 근거: 질병을 그렇게 판단한 이유를 간단히 1문장으로 설명\n"
    "예: \n"
    "질병명: 독감\n"
    "판단 근거: 고열과 오한, 전신통증 증상은 독감과 일치하기 때문.\n"
)

# 5. Qwen 모델 로드 및 응답 생성
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True).eval()

# 응답 생성
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# 프롬프트 이후만 잘라 - 후처리
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = decoded[len(prompt):].strip()

# 중복 방지 간단 후처리: 첫 번째 질병명 ~ 판단 근거까지만 추출
match = re.search(r"질병명:.*?\n판단 근거:.*?(?:\n|$)", response)
if match:
    cleaned_response = match.group().strip()
else:
    cleaned_response = "[오류] 질병명 또는 판단 근거를 찾을 수 없습니다."


# 최종 출력
print("[응답 결과]\n", cleaned_response)