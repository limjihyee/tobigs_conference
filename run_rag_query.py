import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 사용자 질문 예시
user_query = "잇몸이 자주 붓고 피가 나요. 어떤 질병이 의심되나요?"

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
system_prompt = (
    "당신은 의료 질의응답 전문가입니다.\n"
    "사용자의 증상 설명과 유사한 질병 정의들을 참고하여, "
    "가장 가능성 높은 질병명을 추론해주세요.\n"
    "답변은 질병명 하나만, 간결하게 출력해야 합니다.\n\n"
    "예시)\n"
    "사용자 증상: 목이 따갑고 열이 나요.\n\n"
    "정의 목록:\n"
    "[정의 1]\n"
    "질병명: 급성 인두염\n"
    "질문: 인두염의 증상은 어떤가요?\n"
    "답변: 인두염은 인두 부위 염증으로 인한 통증과 열이 동반됩니다.\n\n"
    "[정의 2]\n"
    "질병명: 독감\n"
    "질문: 독감 초기 증상은 무엇인가요?\n"
    "답변: 독감은 전신 근육통과 고열, 오한 등이 대표적인 증상입니다.\n\n"
    "가장 가능성 높은 질병명: 급성 인두염\n"
    "-------------------------"
)

user_prompt = (
    f"사용자 증상: {user_query}\n\n정의 목록:\n"
    "\n".join([f"[정의 {i+1}]\n{docs[I[0][i]]}" for i in range(k)]) +
    "가장 가능성 높은 질병명:"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# 5. Qwen 모델 로드 및 응답 생성
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True).eval()

response = model.chat(tokenizer, messages)

# 6. 출력
print("예측된 질병명:", response)
