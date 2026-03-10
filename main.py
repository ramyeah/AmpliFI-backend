from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import get_rag_response
from app.rag import index, embedding_model, openai_client

app = FastAPI(title="AmpliFI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    user_profile: dict = None

@app.get("/")
def root():
    return {"status": "AmpliFI backend is running"}

@app.post("/ask")
def ask(request: QueryRequest):
    result = get_rag_response(request.query, request.user_profile)
    return result

class QueryRequest(BaseModel):
    query: str
    user_profile: dict = None
    override_prompt: str = None

@app.post("/ask")
def ask(request: QueryRequest):
    result = get_rag_response(request.query, request.user_profile, request.override_prompt)
    return result

class QuizRequest(BaseModel):
    topic: str
    user_profile: dict = None

@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    query_vector = embedding_model.encode(request.topic).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    top_matches = [r for r in results.matches if r.score >= 0.35]
    context = "\n\n".join([m.metadata.get("text", "") for m in top_matches]) if top_matches else "Use your knowledge of Singapore investing."

    name = request.user_profile.get("name", "the user") if request.user_profile else "the user"

    prompt = f"""Generate exactly 5 multiple choice quiz questions about "{request.topic}" for a Singapore university student named {name}.

CONTEXT:
{context}

Respond with ONLY this JSON, no other text:
{{
  "questions": [
    {{
      "question": "Question text?",
      "options": ["A) option", "B) option", "C) option", "D) option"],
      "correct": "A",
      "explanation": "Why A is correct"
    }}
  ]
}}"""

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3
    )

    return {
        "response": completion.choices[0].message.content,
        "disclaimer": False
    }