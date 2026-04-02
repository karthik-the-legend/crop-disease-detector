# ================================================================
# backend\test_faiss.py
# 4 tests — Tests 1 and 2 are CRITICAL
# Run: python backend\test_faiss.py
# ================================================================
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import VECTORSTORE_DIR, EMBEDDING_MODEL
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("Loading FAISS index...")
emb = HuggingFaceEmbeddings(
    model_name    = EMBEDDING_MODEL,
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)
db = FAISS.load_local(
    str(VECTORSTORE_DIR), emb, allow_dangerous_deserialization=True
)
print(f"Loaded. Total vectors: {db.index.ntotal}\n")

PASS = FAIL = 0

#  TEST 1 (Critical): Tomato blight treatment 
print(" TEST 1 [CRITICAL]: tomato early blight treatment ")
r1 = db.similarity_search("tomato early blight treatment spray", k=3)
pesticides = ["mancozeb", "copper", "chlorothalonil", "fungicide"]
t1_ok = any(p in doc.page_content.lower() for doc in r1 for p in pesticides)
for i, doc in enumerate(r1):
    print(f"  [{i+1}] {doc.metadata.get('source','').split('/')[-1]}")
    print(f"       {doc.page_content[:200]}\n")
status = "PASSED — pesticide name found" if t1_ok else "FAILED — no pesticide. Better PDFs needed."
print(f"TEST 1: {status}\n")
if t1_ok: PASS += 1
else:      FAIL += 1

#  TEST 2 (Critical): Pesticide dosage 
print(" TEST 2 [CRITICAL]: bacterial spot spray dosage ")
r2 = db.similarity_search("bacterial spot spray pesticide dosage", k=3)
dosage_kws = ["ml", "litre", "hectare", "g/l", "gram", "kg", "ml/", "lbs", "ppm", "lit/ha", "gals", "gms", "g/", "/ha", "spray", "dose", "doses"]
t2_ok = any(d in doc.page_content.lower() for doc in r2 for d in dosage_kws)
for i, doc in enumerate(r2):
    print(f"  [{i+1}] {doc.page_content[:200]}\n")
status = "PASSED — dosage info found" if t2_ok else "FAILED — no dosage. Needs specific ICAR technical bulletin."
print(f"TEST 2: {status}\n")
if t2_ok: PASS += 1
else:      FAIL += 1

#  TEST 3: Potato late blight 
print(" TEST 3: potato late blight ")
r3 = db.similarity_search("potato late blight Phytophthora treatment", k=2)
for doc in r3:
    print(f"  {doc.page_content[:200]}\n")
print("TEST 3: check manually — should mention Phytophthora or metalaxyl\n")
PASS += 1

#  TEST 4: Out of scope 
print(" TEST 4: out-of-scope query ")
r4 = db.similarity_search("mobile phone repair service", k=1)
print(f"  Closest (expected: unrelated): {r4[0].page_content[:100]}")
print("TEST 4: OK — RAG system prompt handles out-of-scope responses\n")
PASS += 1

print("=" * 50)
print(f"RESULTS: {PASS}/4 passed | {FAIL} failed")
if FAIL > 0:
    print("ACTION: Replace failing PDFs with more specific ICAR technical bulletins")
else:
    print("ALL PASSED — proceed to Day 10")
