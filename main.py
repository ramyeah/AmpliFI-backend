from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.rag import get_rag_response
from app.rag import index, embedding_model, openai_client

app = FastAPI(title="AmpliFI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── /ask ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    user_profile: dict = None
    override_prompt: str = None

@app.get("/")
def root():
    return {"status": "AmpliFI backend is running"}

@app.post("/ask")
def ask(request: QueryRequest):
    result = get_rag_response(request.query, request.user_profile, request.override_prompt)
    return result

# ─── /quiz ────────────────────────────────────────────
class QuizRequest(BaseModel):
    topic: str
    user_profile: dict = None

@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    query_vector = embedding_model.encode(request.topic).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    top_matches = [r for r in results.matches if r.score >= 0.35]
    context = "\n\n".join([m.metadata.get("text", "") for m in top_matches]) if top_matches else "Use your knowledge of Singapore personal finance."
    name = request.user_profile.get("name", "the user") if request.user_profile else "the user"
    prompt = f"""Generate exactly 5 multiple choice quiz questions about "{request.topic}" for a Singapore university student named {name}.
CONTEXT:\n{context}\nRespond with ONLY this JSON, no other text:
{{"questions": [{{"question": "Question text?","options": ["A) option","B) option","C) option","D) option"],"correct": "A","explanation": "Why A is correct"}}]}}"""
    completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.3)
    return {"response": completion.choices[0].message.content, "disclaimer": False}

# ─── /flashcard ───────────────────────────────────────
class FlashcardRequest(BaseModel):
    rag_query: str
    question: str

@app.post("/flashcard")
def get_flashcard_answer(request: FlashcardRequest):
    query_vector = embedding_model.encode(request.rag_query).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    top_matches = [r for r in results.matches if r.score >= 0.35]
    context = "\n\n".join([m.metadata.get("text", "") for m in top_matches]) if top_matches else "Use your knowledge of Singapore personal finance."
    prompt = f"""You are a Singapore financial literacy flashcard assistant. Answer concisely for a university student in Singapore. 2-4 sentences max. Include specific Singapore figures where relevant.\nCONTEXT:\n{context}\nQUESTION: {request.question}\nANSWER:"""
    completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=200, temperature=0.2)
    return {"answer": completion.choices[0].message.content, "disclaimer": False}

# ─── /lesson-section ──────────────────────────────────
class LessonSectionRequest(BaseModel):
    lesson_topic: str
    section_heading: str
    section_key: str
    other_sections: list[str] = []
    user_profile: dict = None

@app.post("/lesson-section")
def get_lesson_section(request: LessonSectionRequest):
    specific_query = f"{request.lesson_topic} {request.section_heading}"
    query_vector = embedding_model.encode(specific_query).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    top_matches = [r for r in results.matches if r.score >= 0.35]
    context = "\n\n".join([m.metadata.get("text", "") for m in top_matches]) if top_matches else "Use your knowledge of Singapore personal finance."
    name = request.user_profile.get("name", "the student") if request.user_profile else "the student"
    other = ", ".join(request.other_sections) if request.other_sections else "none"
    prompt = f"""You are AmpliFI, a Singapore financial literacy educator writing ONE specific section of a lesson.
LESSON TOPIC: {request.lesson_topic}\nTHIS SECTION: {request.section_heading}\nOTHER SECTIONS (DO NOT cover): {other}
Write ONLY about "{request.section_heading}". 150-250 words. Singapore-specific. Address student as "{name}".
Use bold terms, bullets, numbered steps, tables, callouts as appropriate.
CONTEXT:\n{context}\nWrite the "{request.section_heading}" section now:"""
    completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=400, temperature=0.2)
    return {"response": completion.choices[0].message.content, "disclaimer": False}

# ─── /simulate/month ─────────────────────────────────
class SimulateMonthRequest(BaseModel):
    month: int
    income: int
    income_label: str
    bank_balance: float
    savings_balance: float
    credit_card_debt: float
    savings_rate: float
    event_id: str
    event_text: str
    previous_choices: list = []

@app.post("/simulate/month")
def simulate_month(request: SimulateMonthRequest):
    import json, re
    needs = round(request.income * 0.50)
    wants_budget = round(request.income * 0.30)
    savings_target = round(request.income * 0.20)
    disposable = round(request.income - needs)
    months_emergency = round(request.savings_balance / (request.income / 3), 1) if request.income > 0 else 0
    debt_monthly_interest = round(request.credit_card_debt * 0.25 / 12, 2)
    state_context = f"""CHARACTER'S EXACT FINANCIAL STATE:
- Income: ${request.income:,} ({request.income_label}), Month {request.month}/6
- Bank: ${request.bank_balance:,.0f}, Emergency fund: ${request.savings_balance:,.0f} ({months_emergency}mo)
- Debt: ${request.credit_card_debt:,.0f}{f' (${debt_monthly_interest}/mo interest)' if request.credit_card_debt > 0 else ''}
- Budget: Needs ${needs:,} | Wants ${wants_budget:,} | Savings ${savings_target:,}
- Event: {request.event_text}
- History: {chr(10).join([f"  M{c['month']}: {c['bias_label']} ({'✓' if c['was_correct'] else '✗'})" for c in request.previous_choices]) if request.previous_choices else "  First month."}"""
    prompt = f"""{state_context}
Generate 3 options. One correct (is_correct:true), two tempting-but-wrong biases. Use their ACTUAL dollar amounts.
fin_nudge: one sentence surfacing the concept WITHOUT revealing the answer.
Respond ONLY with JSON:
{{"situation_summary":"...","fin_nudge":"...","options":[{{"id":"A","text":"...","bias_label":"...","coin_delta":0,"savings_delta":0,"debt_delta":0,"is_correct":true,"explanation":"..."}}]}}"""
    system_msg = "You are Fin. Respond with valid JSON only — no prose, no markdown."
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":system_msg},{"role":"user","content":prompt}], max_tokens=1200, temperature=0.7)
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: raw = m.group(0)
        data = json.loads(raw)
        if not isinstance(data.get('options'), list) or len(data['options']) < 3:
            raise ValueError("bad options")
        return data
    except Exception as e:
        print(f"simulate_month error: {e}")
        return {"situation_summary": f"${request.bank_balance:,.0f} in bank. {request.event_text}", "fin_nudge": f"With ${disposable:,} disposable, think about which lever to pull first.", "options": [{"id":"A","text":f"Transfer ${savings_target:,} to savings first.","bias_label":"Pay yourself first","coin_delta":15,"savings_delta":savings_target,"debt_delta":0,"is_correct":True,"explanation":"Paying yourself first before spending is the 50/30/20 rule in action."},{"id":"B","text":"Spend freely, save what's left.","bias_label":"Save what's left","coin_delta":-20,"savings_delta":0,"debt_delta":0,"is_correct":False,"explanation":"Saving what's left consistently leaves nothing saved."},{"id":"C","text":f"Put all ${disposable:,} into savings.","bias_label":"All-or-nothing","coin_delta":-5,"savings_delta":disposable,"debt_delta":0,"is_correct":False,"explanation":"Extreme restriction is unsustainable and leads to binge spending."}]}

# ─── /simulate/ask-fin ────────────────────────────────
class AskFinRequest(BaseModel):
    question: str
    month: int
    income: int
    income_label: str
    bank_balance: float
    savings_balance: float
    credit_card_debt: float
    event_text: str
    options_shown: list = []

@app.post("/simulate/ask-fin")
def ask_fin(request: AskFinRequest):
    months_emergency = round(request.savings_balance / (request.income / 3), 1) if request.income > 0 else 0
    options_text = "\n\nOPTIONS (do NOT reveal which is correct):\n" + "\n".join([f"- {o.get('text','')}" for o in request.options_shown]) if request.options_shown else ""
    prompt = f"""Fin, warm Singapore financial advisor. Student state: ${request.income:,}/mo, ${request.bank_balance:,.0f} bank, ${request.savings_balance:,.0f} fund ({months_emergency}mo), event: {request.event_text}{options_text}
Question: "{request.question}"
Answer directly, 3-5 sentences. Use their specific numbers. Don't reveal the correct option. End with a question."""
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=300, temperature=0.4)
        return {"response": completion.choices[0].message.content}
    except:
        return {"response": "I'm having trouble connecting. Think about which principle applies most here — needs vs wants, or debt vs savings?"}

# ─── /simulate/insight ────────────────────────────────
class SimulateInsightRequest(BaseModel):
    income: int
    income_label: str
    start_balance: float
    final_balance: float
    start_coins: int
    final_coins: int
    correct_count: int
    choices: list

@app.post("/simulate/insight")
def simulate_insight(request: SimulateInsightRequest):
    gained = request.final_coins >= request.start_coins
    net_change = request.final_coins - request.start_coins
    choice_lines = "\n".join([f"  M{c['month']} ({c['concept']}): {c['bias_label']} ({'✓' if c['is_correct'] else '✗'})" for c in request.choices])
    wrong = [c for c in request.choices if not c['is_correct']]
    bias_counts = {}
    for c in wrong: bias_counts[c['bias_label']] = bias_counts.get(c['bias_label'], 0) + 1
    top_bias = max(bias_counts, key=bias_counts.get) if bias_counts else None
    prompt = f"""Fin, financial advisor. Write a 3-paragraph personalised insight report (max 200 words).
Student: ${request.income:,}/mo ({request.income_label}), {request.correct_count}/6 correct, balance ${request.start_balance:,.0f}→${request.final_balance:,.0f}{f', top bias: {top_bias}' if top_bias else ''}
Choices:\n{choice_lines}
Para 1: what they did well (specific months). Para 2: pattern noticed (use their numbers). Para 3: one Singapore-specific actionable recommendation.
No bullets. Warm, honest. Sign off "— Fin 🦉" """
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=400, temperature=0.3)
        return {"report": completion.choices[0].message.content}
    except:
        return {"report": f"You made {request.correct_count}/6 correct decisions. Keep practising — every correct call is a habit you're building for real life. — Fin 🦉"}

# ─── /bot-fact ────────────────────────────────────────
class BotFactRequest(BaseModel):
    label: str
    prompt: str

@app.post("/bot-fact")
def get_bot_fact(request: BotFactRequest):
    query_vector = embedding_model.encode(request.prompt).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    top_matches = [r for r in results.matches if r.score >= 0.35]
    context = "\n\n".join([m.metadata.get("text", "") for m in top_matches]) if top_matches else ""
    prompt = f"Answer in 2-3 sentences. Be specific — include real Singapore figures.\nQuestion: {request.label}\nContext: {context}\nAnswer:"
    completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=120, temperature=0.2)
    return {"answer": completion.choices[0].message.content}


# ═══════════════════════════════════════════════════════════════════════
# LIFE SIM — STAGE 1: AI-POWERED GOAL SETTING CONVERSATION
# ═══════════════════════════════════════════════════════════════════════

# ─── /sim/goals/frame-age ─────────────────────────────────────────────
# Called before showing the age chips.
# Fin explains what the retirement age choice actually means in dollar terms.

class SimFrameAgeRequest(BaseModel):
    user_name: str
    income: int
    income_label: str

@app.post("/sim/goals/frame-age")
def sim_frame_age(request: SimFrameAgeRequest):
    print(f"[sim/goals/frame-age] HIT — user={request.user_name} income={request.income}")
    savings_20pct = round(request.income * 0.20)
    prompt = f"""You are Fin, a direct and knowledgeable financial advisor in AmpliFI, a Singapore financial literacy app for university students.

{request.user_name} earns ${request.income:,}/month ({request.income_label}) in their simulation.
At 20% savings, they save ${savings_20pct:,}/month.

Write ONE message (3-4 sentences) that:
1. Opens with something genuinely interesting about retirement age — e.g. the mathematical relationship between retirement age and the size of the number needed. Make it feel like a revelation, not a textbook.
2. Gives ONE concrete example: e.g. "retiring at 45 vs 65 roughly triples the amount you need to save — because you have fewer years to build it AND more years to fund"
3. Ends with a direct question: "When do you want to be financially free?"

Rules:
- Do NOT say "Great question" or "Let's get started" — just open with the insight
- Reference their ${request.income:,}/month income to make it feel personal
- Be direct and honest — not motivational-poster language
- Max 4 sentences"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180, temperature=0.9,
        )
        result = completion.choices[0].message.content.strip()
        print(f"[sim/goals/frame-age] AI response: {result[:80]}")
        return {"response": result}
    except Exception as e:
        print(f"[sim/goals/frame-age] ERROR: {e}")
        return {"response": f"The age you pick changes everything — retiring at 45 vs 65 roughly triples the amount you need, because you have fewer years to build it and more years to fund. On ${request.income:,}/month, starting now makes the difference between those two scenarios. When do you want to be financially free?"}


# ─── /sim/goals/react-age ─────────────────────────────────────────────
# Called after the user picks their retirement age.
# Fin gives a specific, honest reaction to their chosen age.

class SimReactAgeRequest(BaseModel):
    user_name: str
    income: int
    income_label: str
    retire_age: int

@app.post("/sim/goals/react-age")
def sim_react_age(request: SimReactAgeRequest):
    print(f"[sim/goals/react-age] HIT — user={request.user_name} age={request.retire_age}")
    assumed_current_age = 24
    years_to_retire = request.retire_age - assumed_current_age
    savings_20pct = round(request.income * 0.20)

    prompt = f"""You are Fin, a direct financial advisor in AmpliFI.

{request.user_name} just said they want to retire at {request.retire_age}.
Current assumed age: {assumed_current_age}. Years to retire: {years_to_retire}.
Their simulated income: ${request.income:,}/month. 20% savings = ${savings_20pct:,}/month.

Write ONE response (2-3 sentences) that:
1. Gives a specific, honest take on {request.retire_age} as a retirement age — e.g.:
   - Age 40-45: "That's very ambitious — {years_to_retire} years means you need to save aggressively AND invest well to make compounding work for you."
   - Age 50-55: "{years_to_retire} years is a solid runway — enough time for compounding to do most of the heavy lifting if you start now."
   - Age 60-65: "{years_to_retire} years gives you the most time — the tradeoff is you'll need your money to last longer in retirement."
2. Notes ONE specific implication for their income bracket (e.g. how many months/years of saving at ${savings_20pct:,}/month represents meaningful progress)
3. Naturally transitions — does NOT ask the next question (that's handled by the UI)

Rules:
- Be honest, not cheerleader-y. If 40 is unrealistic on their salary, say so gently.
- Use their actual numbers throughout
- Max 3 sentences"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.9,
        )
        result = completion.choices[0].message.content.strip()
        print(f"[sim/goals/react-age] AI response: {result[:80]}")
        return {"response": result}
    except Exception as e:
        print(f"[sim/goals/react-age] ERROR: {e}")
        if request.retire_age <= 45:
            fallback = f"{request.retire_age} is ambitious — {years_to_retire} years means your savings rate and investment returns both need to work hard. At ${savings_20pct:,}/month saved, you'll need strong investment growth to get there."
        elif request.retire_age <= 55:
            fallback = f"{years_to_retire} years is a solid runway — enough for compounding to do most of the heavy lifting if you start now and stay consistent. At ${savings_20pct:,}/month, every year you delay costs you more than you'd expect."
        else:
            fallback = f"{years_to_retire} years gives you the most time — the tradeoff is your money needs to last longer in retirement, so the size of the number matters just as much."
        return {"response": fallback}


# ─── /sim/goals/frame-lifestyle ───────────────────────────────────────
# Called before showing the lifestyle cost builder.
# Fin explains why people underestimate retirement costs — and which categories matter most.

class SimFrameLifestyleRequest(BaseModel):
    user_name: str
    income: int
    retire_age: int

@app.post("/sim/goals/frame-lifestyle")
def sim_frame_lifestyle(request: SimFrameLifestyleRequest):
    print(f"[sim/goals/frame-lifestyle] HIT — user={request.user_name}")
    prompt = f"""You are Fin, a direct financial advisor in AmpliFI (Singapore).

{request.user_name} wants to retire at {request.retire_age}. Income: ${request.income:,}/month.

Write ONE message (3-4 sentences) that:
1. Explains WHY people almost always underestimate their retirement number — the two biggest surprises are healthcare costs (which balloon after 60) and the fact that you'll have more time to spend money, not less
2. Calls out one Singapore-specific cost reality — e.g. HDB mortgage may be paid off, but service charges, conservancy, and property tax still apply; or MediShield Life covers hospitalisation but not everything
3. Ends with a direct instruction: "Let's build your number category by category. Tap each one to set your expected monthly spend."

Rules:
- Be specific — use real Singapore figures where possible (e.g. "healthcare for a Singaporean over 65 averages $400–$800/month out of pocket")
- Do NOT be preachy. Be informative like a knowledgeable friend.
- Max 4 sentences"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180, temperature=0.9,
        )
        result = completion.choices[0].message.content.strip()
        print(f"[sim/goals/frame-lifestyle] AI response: {result[:80]}")
        return {"response": result}
    except Exception as e:
        print(f"[sim/goals/frame-lifestyle] ERROR: {e}")
        return {"response": "Most people underestimate their FI Number by 30–40% — the two surprises are healthcare (which can run $400–$800/month out of pocket after 60 in Singapore) and the simple fact that you have more free time, so more opportunity to spend. Your HDB mortgage may be paid off, but conservancy charges, utilities, and medical bills don't stop. Let's build your number category by category — tap each one to set your expected monthly spend."}


# ─── /sim/goals/react-ffn ─────────────────────────────────────────────
# Called after the lifestyle cost builder produces a monthly total + FFN.
# Fin reacts to the FFN — confirms it, challenges it if unrealistic, explains the math.

class SimReactFfnRequest(BaseModel):
    user_name: str
    income: int
    income_label: str
    retire_age: int
    monthly_total: int       # sum of all lifestyle cost categories
    ffn: int                 # monthly_total * 12 / 0.04
    is_realistic: bool       # frontend determines: ffn / (years * 12 * income * 0.2) > 10x = unrealistic

@app.post("/sim/goals/react-ffn")
def sim_react_ffn(request: SimReactFfnRequest):
    assumed_age = 24
    years = request.retire_age - assumed_age
    savings_20pct = round(request.income * 0.20)
    # Pure savings (no investment returns) would take this many years
    pure_savings_years = round(request.ffn / (savings_20pct * 12), 1) if savings_20pct > 0 else 999
    # With ~6% annualised investment returns, rough FV estimate
    # FV = PMT * ((1+r)^n - 1) / r
    import math
    r_monthly = 0.06 / 12
    n_months = years * 12
    if r_monthly > 0 and n_months > 0:
        fv_invested = savings_20pct * ((1 + r_monthly)**n_months - 1) / r_monthly
    else:
        fv_invested = savings_20pct * n_months
    reaches_ffn = fv_invested >= request.ffn
    shortfall = max(0, request.ffn - fv_invested)

    prompt = f"""You are Fin, a direct and honest financial advisor in AmpliFI (Singapore).

{request.user_name}'s FI Number: ${request.ffn:,}
Breakdown: ${request.monthly_total:,}/month × 12 months × 25 (the 4% rule) = ${request.ffn:,}
Their income: ${request.income:,}/month ({request.income_label})
Saving 20% = ${savings_20pct:,}/month
Years to retire: {years} (target age {request.retire_age})
Pure savings (no investing): would take {pure_savings_years} years — {'faster than their timeline' if pure_savings_years <= years else 'SLOWER than their timeline'}
With ~6% annual investment returns over {years} years: ~${fv_invested:,.0f} — {'✓ reaches FI Number' if reaches_ffn else f'✗ ${shortfall:,.0f} short of FI Number'}

Write ONE honest, specific response (4-5 sentences) that:
1. States their FFN clearly and what it means in plain English
2. {'Confirms this is achievable and explains specifically how (investing the ${savings_20pct:,}/month at ~6% gets them there)' if reaches_ffn else f'Honestly flags that saving alone is not enough — they need investment returns, AND/OR a higher savings rate. Be specific about the shortfall: ${shortfall:,.0f}'}
3. {'If the monthly total seems high (> $5,000/month for a student income), gently ask if all categories are realistic — e.g. $1,200/month on housing assumes HDB is paid off, not that they are renting' if request.monthly_total > 4000 else 'Affirms their lifestyle estimate looks reasonable for Singapore'}
4. Ends by asking: "Does this feel right, or do you want to adjust any of your cost estimates?"

Rules:
- Be honest. If the number is hard to reach, say so — but show the path (higher savings rate, investing, starting now)
- Use ONLY their actual numbers — no generic examples
- Max 5 sentences"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220, temperature=0.85,
        )
        return {
            "response": completion.choices[0].message.content.strip(),
            "reaches_ffn": reaches_ffn,
            "fv_invested": round(fv_invested),
            "shortfall": round(shortfall),
            "pure_savings_years": pure_savings_years,
        }
    except Exception as e:
        fallback = f"Your FI Number is ${request.ffn:,} — that's ${request.monthly_total:,}/month × 12 × 25, the amount you need invested to live off returns indefinitely. "
        if reaches_ffn:
            fallback += f"Saving ${savings_20pct:,}/month and investing it at ~6% annual returns over {years} years gets you to approximately ${fv_invested:,.0f} — enough. The key is starting now and staying consistent. Does this feel right?"
        else:
            fallback += f"Here's the honest reality: saving ${savings_20pct:,}/month alone isn't enough — even with investment returns, you'd be about ${shortfall:,.0f} short. You'd need either a higher savings rate or to adjust your retirement lifestyle. Does this feel right, or do you want to adjust your cost estimates?"
        return {"response": fallback, "reaches_ffn": reaches_ffn, "fv_invested": round(fv_invested), "shortfall": round(shortfall), "pure_savings_years": pure_savings_years}


# ─── /sim/goals/react-goal ────────────────────────────────────────────
# Called after the user sets their short-term goal label + amount.
# Fin reacts with context and bridges to the summary.

class SimReactGoalRequest(BaseModel):
    user_name: str
    income: int
    goal_label: str
    goal_amount: int

@app.post("/sim/goals/react-goal")
def sim_react_goal(request: SimReactGoalRequest):
    import math
    savings_20pct = round(request.income * 0.20)
    months_to_goal = math.ceil(request.goal_amount / savings_20pct) if savings_20pct > 0 else 0

    prompt = f"""You are Fin, a direct financial advisor in AmpliFI (Singapore).

{request.user_name} wants to save ${request.goal_amount:,} for: "{request.goal_label}"
Their income: ${request.income:,}/month. 20% savings = ${savings_20pct:,}/month.
At that rate: {months_to_goal} months to reach this goal.

Write ONE response (2-3 sentences) that:
1. Gives ONE specific, genuine observation about this goal — tailored to what it actually is:
   - Laptop/tech: it's an investment in earning capacity, not just a purchase
   - Trip home/travel: acknowledge the real value of connection and rest
   - Emergency fund: the single highest-ROI financial move available to them right now
   - Course/education: same as laptop — human capital investment
   - Anything else: find something honest and specific to say
2. States the timeline concretely: "At ${savings_20pct:,}/month, you'd reach ${request.goal_amount:,} in {months_to_goal} months"
3. Does NOT ask another question — signals that the blueprint is ready

Rules:
- Max 3 sentences
- Sound like a real person. Vary the opener — do NOT start with "Great!" or "That's a great goal!"
- Be specific to what the goal actually is"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=140, temperature=0.9,
        )
        return {"response": completion.choices[0].message.content.strip(), "months_to_goal": months_to_goal}
    except Exception as e:
        return {"response": f"At ${savings_20pct:,}/month saved, you'd reach ${request.goal_amount:,} in about {months_to_goal} months — less than a year if you stay consistent. Here's your complete financial blueprint.", "months_to_goal": months_to_goal}