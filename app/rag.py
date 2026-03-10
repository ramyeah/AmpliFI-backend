import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# Initialise clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_rag_response(query: str, user_profile: dict = None, override_prompt: str = None) -> dict:
    # Step 1: Embed the query
    query_vector = embedding_model.encode(query).tolist()

    # Step 2: Search Pinecone for top-3 relevant chunks
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    print(f"Top scores: {[r.score for r in results.matches]}")

    # Step 3: Check similarity threshold
    top_matches = [r for r in results.matches if r.score >= 0.35]

    # Step 4: Build context
    if top_matches:
        context = "\n\n".join([match.metadata.get("text", "") for match in top_matches])
    else:
        context = "Use your general knowledge about Singapore financial products and investing."

    # Step 5: Build personalisation string
    personalisation = ""
    if user_profile:
        personalisation = f"""
The user's profile:
- Name: {user_profile.get('name', 'the user')}
- Age: {user_profile.get('age', 'unknown')}
- Income bracket: {user_profile.get('income', 'unknown')}
- Family status: {user_profile.get('familyStatus', 'unknown')}
- Financial goal: {user_profile.get('goal', 'unknown')}
"""

    # Step 6: Use override prompt if provided, otherwise use default
    if override_prompt:
        prompt = f"{override_prompt}\n\nCONTEXT (use if relevant):\n{context}"
    else:
        prompt = f"""You are AmpliFI, a friendly Singapore financial literacy assistant.
Answer the user's question using the context provided below and your knowledge of Singapore finance.
Always use Singapore-specific examples, dollar amounts in SGD, and reference
relevant schemes like CPF, MAS regulations, or local investment products where appropriate.
Keep responses concise, educational, and suitable for a university student.

{personalisation}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )

    return {
        "response": completion.choices[0].message.content,
        "disclaimer": False,
        "sources_used": len(top_matches)
    }