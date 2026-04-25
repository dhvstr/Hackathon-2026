import streamlit as st
import json
import fitz  # PyMuPDF
import requests
import google.generativeai as genai
from datetime import datetime

st.set_page_config(page_title="SME Financial Health Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    .metric-label { font-size: 0.85rem; color: #94a3b8; }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: #f1f5f9; }
    .metric-delta { font-size: 0.8rem; }
    .insight-box {
        background: #0f172a;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# Placeholder Functions 

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, Exception):
    GEMINI_API_KEY = ""

try:
    ZAI_API_KEY = st.secrets["ZAI_API_KEY"]
except (KeyError, Exception):
    ZAI_API_KEY = ""
ZAI_API_URL = "https://api.ilmu.ai/v1/chat/completions"

if "zai_key" not in st.session_state:
    st.session_state.zai_key = ZAI_API_KEY

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = GEMINI_API_KEY


def _enrich_insight(parsed: dict, cash) -> dict:
    """Normalize AI-parsed insight: convert bare int confidence_score to
    display format and compute quantifiable_impact_rm locally if missing."""
    import re as _re
    if isinstance(parsed.get("confidence_score"), int):
        parsed["confidence_score"] = f"Confidence: {parsed['confidence_score']}%"
    if "quantifiable_impact_rm" not in parsed:
        conf_val = parsed.get("confidence_score", "50%")
        _m = _re.search(r"(\d+)", str(conf_val))
        conf_int = int(_m.group(1)) if _m else 50
        risk_pct = conf_int / 100
        if cash and cash > 0:
            parsed["quantifiable_impact_rm"] = f"RM {round(cash * risk_pct):,} — {risk_pct:.0%} of RM {cash:,} cash at risk"
        else:
            parsed["quantifiable_impact_rm"] = f"Cannot calculate without cash data. Risk level: {conf_int}%"
    return parsed


def generate_ai_insight(headline: str) -> dict:
    """Decision Intelligence Engine — tries Gemini first, then Z.AI,
    then local fallback. Produces a CFO-level, context-aware financial
    recommendation with localized Malaysian economic intelligence and
    non-linear risk assessment.

    Returns a dict with keys: action_recommendation, clear_explanation,
    confidence_score, quantifiable_impact_rm.
    """
    cash = st.session_state.get("cash_balance")
    revenue = st.session_state.get("monthly_revenue")
    cash_str = f"RM {cash:,.0f}" if cash else "Not provided — assume unknown liquidity"
    revenue_str = f"RM {revenue:,.0f}" if revenue else "Not provided — assume unknown revenue"

    system_prompt = (
       "Act as a world-class CFO focusing on Malaysian SMEs. Analyze headlines with local context (BNM policy, SST, Ringgit trends, trade ties, SME schemes). Tailor advice to the business's cash and revenue — strategies differ for low vs. high liquidity, exporters vs. importers, borrowers vs. savers. Assess risk dynamically based on source reliability,"
       "market volatility, relevance, and data completeness. "
       "CRITICAL OUTPUT RULES: Return strictly valid JSON only. Do NOT wrap output in markdown code blocks (no ```json). "
       "Return a JSON object with these keys: 'action_recommendation' (max 40 words), 'clear_explanation' (max 40 words), and 'confidence_score' (int 0-100). "
       "DO NOT include a 'quantitative_analysis' or 'quant' section. End the JSON immediately after the confidence score."
    )

    user_prompt = (
        f'News headline: "{headline}"\n\n'
        f"Business current metrics:\n"
        f"- Cash Balance: {cash_str}\n"
        f"- Monthly Revenue: {revenue_str}\n\n"
        "As the CFO, analyze the specific economic implications of this headline for this Malaysian SME. "
        "Apply localized intelligence, non-linear thinking, and dynamic risk assessment. "
        "Return strictly valid JSON (no markdown fences) with exactly three keys: "
        "action_recommendation (max 40 words), clear_explanation (max 40 words), confidence_score (int 0-100). "
        "End the JSON immediately after confidence_score."
    )

    # Gemini 
    gemini_key = st.session_state.get("gemini_key", "")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key, client_options={"api_endpoint": "generativelanguage.googleapis.com"})
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_prompt
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                    stop_sequences=["}\n"]
                )
            )
            raw = response.text
            if raw:
                parsed = _robust_json_parse(raw)
                if parsed and "action_recommendation" in parsed:
                    return _enrich_insight(parsed, cash)
                    return parsed
                st.warning("Gemini returned unexpected format. Trying Z.AI...")
                with st.expander("Debug: Raw Gemini Response"):
                    st.text(raw)
            else:
                st.warning("Gemini returned an empty response. Trying Z.AI...")
        except Exception as e:
            st.warning(f"Gemini API call failed ({e}). Trying Z.AI...")

    # Z.ai GLM
    zai_key = st.session_state.get("zai_key", "")
    if zai_key:
        try:
            payload = {
                "model": "ilmu-glm-5.1",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            headers = {
                "Authorization": f"Bearer {zai_key}",
                "Content-Type": "application/json"
            }
            resp = requests.post(ZAI_API_URL, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            if raw:
                parsed = _robust_json_parse(raw)
                if parsed and "action_recommendation" in parsed:
                    return _enrich_insight(parsed, cash)
                st.warning("Z.AI GLM returned unexpected format. Falling back to local analysis.")
                with st.expander("Debug: Raw Z.AI Response"):
                    st.text(raw)
            else:
                st.warning("Z.AI GLM returned an empty response. Falling back to local analysis.")
        except Exception as e:
            st.warning(f"Z.AI GLM API call failed ({e}). Falling back to local analysis.")

    # Local fallback
    return _local_insight_fallback(headline, cash, revenue)


def _local_insight_fallback(headline: str, cash, revenue) -> dict:
    """Local calculation-based fallback when no AI API is available.
    Applies non-linear logic: recommendations change based on the user's
    actual cash and revenue position, not one-size-fits-all.
    Returns the same 4-key structure as the AI prompt.
    """
    import re

    h = headline.lower()
    cash_num = cash if cash and cash > 0 else None
    rev_num = revenue if revenue and revenue > 0 else None

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", headline)
    pct = float(pct_match.group(1)) / 100 if pct_match else 0.03


    def cash_tier():
        if cash_num is None:
            return "unknown"
        # Use revenue to refine: a business with RM 5K cash but RM 50K revenue
        # is less fragile than one with RM 5K cash and RM 5K revenue
        monthly_buffer = rev_num if rev_num else 0
        effective_reserve = cash_num + (monthly_buffer * 2)  # 2 months revenue as buffer
        if effective_reserve < 20000:
            return "survival"      
        elif effective_reserve < 150000:
            return "stable"        
        else:
            return "growth"       

    tier = cash_tier()

   
    if any(w in h for w in ["ringgit", "currency", "forex", "usd", "dollar"]):
        if tier == "survival":
            action = (
                "Immediately convert any USD receivables to Ringgit at current rates — "
                "you cannot afford further depreciation with limited reserves. Freeze all USD-denominated purchases."
            )
            explanation = (
                f"With only RM {cash_num:,} in cash, a further {pct:.0%} Ringgit slide would erase "
                f"RM {round(cash_num * pct):,} of purchasing power — nearly unaffordable for a survival-tier SME. "
                f"BNM's current stance on Ringgit intervention makes speculative waiting too risky at this cash level. "
                f"Lock in what you have now rather than gambling on recovery."
            )
            conf = 0.82
        elif tier == "stable":
            action = (
                "Hedge 50% of USD exposure with forward contracts and keep 50% unhedged to benefit "
                "if Ringgit recovers. Accelerate USD receivable collections but do not freeze all imports."
            )
            explanation = (
                f"With RM {cash_num:,} in cash, a {pct:.0%} Ringgit move means RM {round(cash_num * pct):,} at risk — "
                f"significant but not catastrophic. A partial hedge preserves upside if BNM intervention stabilizes the Ringgit, "
                f"while the hedged half protects against further erosion. "
                f"This balanced approach fits a business with moderate but not unlimited reserves."
            )
            conf = 0.75
        elif tier == "growth":
            action = (
                "Use this as a buying opportunity: lock in forward contracts for 6 months of USD payables "
                "at the weaker Ringgit rate. If importing, bulk-buy now before further depreciation."
            )
            explanation = (
                f"With RM {cash_num:,} in reserves, a {pct:.0%} Ringgit shift (RM {round(cash_num * pct):,} paper loss) "
                f"is absorbable. Growth-tier SMEs can exploit FX volatility by locking in favorable forward rates "
                f"and front-loading imports at the current rate before BNM policy shifts. "
                f"The stronger balance sheet turns currency risk into a competitive advantage."
            )
            conf = 0.78
        else:
            action = "Convert all foreign currency holdings to Ringgit immediately and halt USD-denominated purchases until stability returns."
            explanation = (
                f"Without visibility into your cash position, assume maximum FX exposure. "
                f"Ringgit volatility of {pct:.0%} against USD directly impacts import costs and export revenues. "
                f"Enter your Cash Balance in the Cash Flow page for a tailored strategy."
            )
            conf = 0.40

        if cash_num:
            impact_str = f"RM {round(cash_num * pct):,} — {pct:.0%} of RM {cash_num:,} cash at FX risk"
        else:
            impact_str = f"Cannot calculate without cash data. Assume {pct:.0%} FX exposure on all foreign transactions"

        return {
            "action_recommendation": action,
            "clear_explanation": explanation,
            "confidence_score": f"Confidence: {int(conf * 100)}%",
            "quantifiable_impact_rm": impact_str
        }

   
    elif any(w in h for w in ["tax", "gst", "sst", "levy"]):
        if tier == "survival":
            action = (
                "Defer ALL non-essential spending until the tax change takes effect. "
                "If SST is increasing, pre-purchase critical supplies now before the hike hits. "
                "If SST is decreasing, delay every purchase possible to capture the lower rate."
            )
            explanation = (
                f"With only RM {cash_num:,} in cash, a {pct:.0%} SST shift changes your cost base by "
                f"RM {round(cash_num * pct):,} — the difference between surviving the quarter or not. "
                f"At this cash level, timing purchases around the tax effective date is not optional, it is survival. "
                f"Every RM saved on tax is a RM that extends your runway."
            )
            conf = 0.85
        elif tier == "stable":
            action = (
                "Accelerate capital expenditure before the tax increase to claim current-rate deductions, "
                "but keep a 60-day cash reserve untouched. Schedule remaining purchases after the new rate."
            )
            explanation = (
                f"RM {cash_num:,} in cash gives room to time purchases strategically. "
                f"A {pct:.0%} SST change means RM {round(cash_num * pct):,} in shifted costs. "
                f"Accelerating deductible expenses before the hike and deferring non-critical spending after "
                f"optimizes your effective tax rate without jeopardizing liquidity. "
                f"Unlike survival-tier businesses, you can afford to be strategic rather than reactive."
            )
            conf = 0.76
        elif tier == "growth":
            action = (
                "Consult a tax advisor within 48 hours to restructure procurement and maximize input tax credits. "
                "Consider accelerating expansion-related purchases to lock in current tax treatment."
            )
            explanation = (
                f"With RM {cash_num:,} in reserves, the absolute RM {round(cash_num * pct):,} impact of a {pct:.0%} "
                f"SST shift is large enough to warrant professional tax restructuring. "
                f"Growth-tier businesses should focus on input tax credit optimization and timing large capital purchases, "
                f"not just deferring small expenses. The payoff from proper restructuring far exceeds advisory fees."
            )
            conf = 0.73
        else:
            action = "Determine whether the tax change is an increase or decrease, then time all purchases accordingly. Enter your cash balance for a tailored strategy."
            explanation = (
                f"Without cash data, a {pct:.0%} SST change has unquantifiable impact. "
                f"The direction of the change determines the strategy: increase = pre-buy, decrease = delay. "
                f"Provide your Cash Balance and Revenue for a specific RM calculation."
            )
            conf = 0.35

        if cash_num:
            impact_str = f"RM {round(cash_num * pct):,} — {pct:.0%} SST shift on RM {cash_num:,} cash"
        else:
            impact_str = "Cannot calculate without cash data"

        return {
            "action_recommendation": action,
            "clear_explanation": explanation,
            "confidence_score": f"Confidence: {int(conf * 100)}%",
            "quantifiable_impact_rm": impact_str
        }

   
    elif any(w in h for w in ["interest", "rate", "bnm", "opr"]):
        debt_assumed = round(cash_num * 0.5) if cash_num else None

        if tier == "survival":
            action = (
                "If you have any variable-rate debt, refinance to fixed IMMEDIATELY — "
                "even a small OPR hike on your debt load could break cash flow. "
                "If rates are falling, do NOT take on new debt; use the savings to build reserves."
            )
            explanation = (
                f"With RM {cash_num:,} in cash, an OPR-linked rate increase of {pct:.0%} on an estimated "
                f"RM {debt_assumed:,} in debt means RM {round(debt_assumed * pct):,} in extra interest per cycle. "
                f"At survival-tier, this is the difference between making payroll or not. "
                f"BNM's OPR decisions ripple through all variable-rate facilities — lock in fixed rates now "
                f"before the next Monetary Policy Committee meeting."
            )
            conf = 0.80
        elif tier == "stable":
            action = (
                "Refinance 70% of variable-rate debt to fixed and keep 30% variable to benefit "
                "if rates drop. Simultaneously negotiate for a rate floor on the fixed portion."
            )
            explanation = (
                f"With RM {cash_num:,} in cash and ~RM {debt_assumed:,} in assumed debt, "
                f"a {pct:.0%} OPR shift changes interest costs by RM {round(debt_assumed * pct):,}. "
                f"A partial fixed-variable split hedges against both directions — "
                f"if BNM cuts OPR, the variable portion benefits; if OPR rises, the fixed portion shields you. "
                f"This suits a business with enough reserves to absorb moderate rate moves."
            )
            conf = 0.72
        elif tier == "growth":
            action = (
                "Lock in long-term fixed-rate financing now at current OPR levels before any hike. "
                "Use the rate environment to negotiate favorable terms — lenders compete for well-capitalized borrowers."
            )
            explanation = (
                f"With RM {cash_num:,} in reserves, you are an attractive borrower. "
                f"A {pct:.0%} OPR change on ~RM {debt_assumed:,} in debt equals RM {round(debt_assumed * pct):,} in shifted interest. "
                f"Growth-tier SMEs should use current rate conditions to secure long-term fixed financing — "
                f"the cost of waiting (potential OPR hike) outweighs the benefit of a possible cut. "
                f"Negotiate from strength while lenders see your balance sheet as low-risk."
            )
            conf = 0.77
        else:
            action = "Refinance all variable-rate loans to fixed immediately if rates are expected to rise. Enter your cash balance for a calculated strategy."
            explanation = (
                f"Without cash data, assume {pct:.0%} OPR impact on all variable-rate debt. "
                f"BNM's OPR direction determines whether to lock in or stay variable. "
                f"Provide your Cash Balance and Revenue for a position-specific recommendation."
            )
            conf = 0.38

        if debt_assumed:
            impact_str = f"RM {round(debt_assumed * pct):,} — {pct:.0%} OPR shift on ~RM {debt_assumed:,} estimated debt"
        else:
            impact_str = "Cannot calculate without cash data. Assume 50% debt-to-cash ratio"

        return {
            "action_recommendation": action,
            "clear_explanation": explanation,
            "confidence_score": f"Confidence: {int(conf * 100)}%",
            "quantifiable_impact_rm": impact_str
        }

   
    else:
        gen_pct = 0.02  

       
        if any(w in h for w in ["inflation", "price hike", "price increase", "naik harga", "kos sara hidup", "cost of living"]):
            if tier == "survival":
                action = (
                    "Renegotiate all supplier contracts immediately — lock in current prices for 6 months. "
                    "Switch to lower-cost alternatives for non-critical inputs and eliminate discretionary spending."
                )
                explanation = (
                    f"Inflation erodes the real value of RM {cash_num:,} in cash. "
                    f"At a {pct:.0%} inflation rate, purchasing power drops by RM {round(cash_num * pct):,} annually. "
                    f"For a survival-tier SME, this means fixed costs will consume more of the same revenue — "
                    f"locking in prices now is more important than waiting for government intervention."
                ) if cash_num else (
                    f"Inflation of {pct:.0%} erodes purchasing power across all operations. "
                    f"Without cash data, assume cost escalation of 3–5% on recurring expenses."
                )
                conf = 0.72
            elif tier == "stable":
                action = (
                    "Negotiate 12-month fixed-price contracts with key suppliers. "
                    "Build a 10% cost buffer into your budget and review pricing strategy to pass on manageable cost increases."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a {pct:.0%} inflation impact means RM {round(cash_num * pct):,} "
                    f"in annual cost escalation. Stable-tier SMEs should lock in supplier prices now "
                    f"while they still have leverage, and selectively pass cost increases to customers."
                ) if cash_num else "Inflation pressures require immediate supplier price negotiations."
                conf = 0.68
            elif tier == "growth":
                action = (
                    "Use cash reserves to bulk-purchase inventory at current prices before inflation hits. "
                    "Consider acquiring smaller competitors who cannot absorb the cost pressure."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, inflation of {pct:.0%} (RM {round(cash_num * pct):,} annual erosion) "
                    f"is manageable. Growth-tier SMEs can front-load inventory purchases and use their cost advantage "
                    f"to gain market share while smaller competitors struggle."
                ) if cash_num else "Inflation creates acquisition opportunities for well-capitalized businesses."
                conf = 0.70
            else:
                action = "Lock in supplier prices immediately and build a cost buffer into all budgets. Enter your cash balance for a tailored strategy."
                explanation = f"Inflation of {pct:.0%} will increase operating costs across the board. Negotiate fixed-price contracts now."
                conf = 0.45

            if cash_num:
                impact_str = f"RM {round(cash_num * pct):,} — {pct:.0%} inflation erosion on RM {cash_num:,} cash"
            else:
                impact_str = f"Cannot calculate without cash data. Assume {pct:.0%} cost escalation"

            return {
                "action_recommendation": action,
                "clear_explanation": explanation,
                "confidence_score": f"Confidence: {int(conf * 100)}%",
                "quantifiable_impact_rm": impact_str
            }

        
        elif any(w in h for w in ["supply chain", "shortage", "tariff", "trade war", "import ban", "export restriction", "sembako", "kekurangan"]):
            if tier == "survival":
                action = (
                    "Immediately identify alternative local suppliers for all critical inputs. "
                    "Build a 60-day inventory buffer for top 5 materials. Do NOT rely on single-source suppliers."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a supply disruption that increases input costs by even 10% "
                    f"means RM {round(cash_num * 0.10):,} in additional expenses you cannot absorb. "
                    f"Survival-tier SMEs must diversify suppliers NOW — a single disrupted supply line "
                    f"could halt operations entirely."
                ) if cash_num else (
                    "Supply chain disruptions threaten operations. Diversify suppliers immediately "
                    "and build inventory buffers for critical materials."
                )
                conf = 0.70
            elif tier == "stable":
                action = (
                    "Diversify to 2–3 suppliers per critical input, including at least one local source. "
                    "Negotiate buffer stock arrangements with existing suppliers at current prices."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, supply chain disruption poses a moderate risk. "
                    f"A 10% input cost increase would add RM {round(cash_num * 0.10):,} in expenses — "
                    f"uncomfortable but survivable. Dual-sourcing now prevents single-point failure."
                ) if cash_num else "Supply chain risks require dual-sourcing and inventory buffers."
                conf = 0.65
            elif tier == "growth":
                action = (
                    "Pre-purchase 6 months of critical inventory at current prices. "
                    "Explore vertical integration or exclusive supplier agreements to lock in supply."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, supply disruption is an opportunity. "
                    f"Pre-purchasing inventory at current prices hedges against tariff-driven cost increases, "
                    f"while competitors without reserves will face stockouts. "
                    f"Exclusive supplier agreements can lock in preferential pricing."
                ) if cash_num else "Well-capitalized businesses should pre-purchase inventory and negotiate exclusive supply deals."
                conf = 0.67
            else:
                action = "Identify alternative suppliers and build inventory buffers. Enter your cash balance for a specific strategy."
                explanation = "Supply chain disruptions increase costs and risk stockouts. Diversify suppliers now."
                conf = 0.40

            if cash_num:
                impact_str = f"RM {round(cash_num * 0.10):,} — estimated 10% input cost increase from supply disruption"
            else:
                impact_str = "Cannot calculate without cash data. Assume 5–10% input cost increase"

            return {
                "action_recommendation": action,
                "clear_explanation": explanation,
                "confidence_score": f"Confidence: {int(conf * 100)}%",
                "quantifiable_impact_rm": impact_str
            }

        elif any(w in h for w in ["minimum wage", "wage", "salary", "gaji", "pekerja", "worker", "labor", "labour", "employment"]):
            wage_impact_pct = pct if pct > 0 else 0.05 
            if tier == "survival":
                action = (
                    "Audit all labor costs immediately. Identify roles that can be automated or outsourced. "
                    "Delay all new hires. Consider shifting to contract/freelance workers for non-core functions."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a {wage_impact_pct:.0%} wage increase on your payroll "
                    f"could add RM {round(cash_num * wage_impact_pct * 0.3):,} in monthly costs (assuming 30% payroll ratio). "
                    f"At survival-tier, even small payroll increases threaten the runway. "
                    f"Automate or outsource non-core tasks before the wage change takes effect."
                ) if cash_num else (
                    f"A {wage_impact_pct:.0%} wage increase directly impacts monthly operating costs. "
                    f"Audit labor costs and identify automation opportunities immediately."
                )
                conf = 0.68
            elif tier == "stable":
                action = (
                    "Review staffing levels and cross-train employees for multiple roles. "
                    "Invest in productivity tools to offset higher per-employee costs. "
                    "Plan hiring freezes for the next quarter."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a {wage_impact_pct:.0%} wage increase adds an estimated "
                    f"RM {round(cash_num * wage_impact_pct * 0.3):,} in monthly payroll costs. "
                    f"Stable-tier SMEs should offset this by investing in productivity improvements "
                    f"rather than reducing headcount — cross-training and automation preserve capacity."
                ) if cash_num else "Wage increases raise operating costs. Invest in productivity to offset."
                conf = 0.63
            elif tier == "growth":
                action = (
                    "Use the wage change as a retention opportunity — offer above-minimum packages to attract "
                    "top talent from competitors who are cutting back. Invest in training to boost output per employee."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, a {wage_impact_pct:.0%} wage increase "
                    f"(~RM {round(cash_num * wage_impact_pct * 0.3):,} additional monthly payroll) is absorbable. "
                    f"Growth-tier SMEs can flip the wage change into a competitive advantage by attracting "
                    f"talent from weaker competitors who cannot afford to retain staff."
                ) if cash_num else "Wage changes create talent acquisition opportunities for well-capitalized businesses."
                conf = 0.60
            else:
                action = "Audit labor costs and prepare for increased payroll expenses. Enter your cash balance for a tailored strategy."
                explanation = f"Wage policy changes directly affect monthly operating costs. Prepare adjusted budgets."
                conf = 0.38

            if cash_num:
                impact_str = f"RM {round(cash_num * wage_impact_pct * 0.3):,}/month — {wage_impact_pct:.0%} wage increase on ~30% payroll ratio of RM {cash_num:,}"
            else:
                impact_str = "Cannot calculate without cash data. Assume 5% wage increase on payroll"

            return {
                "action_recommendation": action,
                "clear_explanation": explanation,
                "confidence_score": f"Confidence: {int(conf * 100)}%",
                "quantifiable_impact_rm": impact_str
            }

        
        elif any(w in h for w in ["digital", "ai", "automation", "technology", "e-commerce", "fintech", "e-dagang"]):
            if tier == "survival":
                action = (
                    "Adopt free or low-cost digital tools immediately — e-invoicing, digital payments, "
                    "and basic accounting software. Do NOT invest in custom solutions; use off-the-shelf SaaS."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, large tech investments are not viable. "
                    f"However, free tools like LHDN e-invoicing and basic POS systems can reduce processing costs "
                    f"by 10–15%, saving RM {round(cash_num * 0.10):,} annually. "
                    f"Focus on compliance tools first — LHDN mandates e-invoicing for all SMEs."
                ) if cash_num else (
                    "Digital transformation is essential but must be low-cost. Use free SaaS tools for compliance and payments."
                )
                conf = 0.58
            elif tier == "stable":
                action = (
                    "Allocate 3–5% of revenue to digital adoption. Prioritize e-invoicing compliance, "
                    "automated bookkeeping, and online sales channels to expand reach without proportional cost increase."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, investing RM {round(cash_num * 0.04):,} "
                    f"(4% of cash) in digital tools can reduce manual processing costs by 20–30% "
                    f"and open new revenue channels. Start with compliance tools and expand to revenue-generating tech."
                ) if cash_num else "Invest 3–5% of revenue in digital adoption. Prioritize compliance and automation."
                conf = 0.60
            elif tier == "growth":
                action = (
                    "Build a dedicated digital strategy. Invest in AI-powered analytics, "
                    "full ERP integration, and e-commerce platforms. Consider acquiring digital-native competitors."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, a comprehensive digital transformation "
                    f"is affordable and yields high ROI. Growth-tier SMEs that invest in AI and automation "
                    f"can reduce operating costs by 20–40% while scaling revenue without proportional headcount increase."
                ) if cash_num else "Comprehensive digital transformation is the highest-ROI investment for growth businesses."
                conf = 0.62
            else:
                action = "Start with free digital tools for compliance and payments. Enter your cash balance for a tailored investment strategy."
                explanation = "Digital adoption is essential for SME competitiveness. Start with low-cost SaaS tools."
                conf = 0.35

            if cash_num:
                impact_str = f"RM {round(cash_num * 0.10):,}/year — estimated 10% cost reduction from digital adoption"
            else:
                impact_str = "Cannot calculate without cash data. Assume 10–15% cost reduction from digital tools"

            return {
                "action_recommendation": action,
                "clear_explanation": explanation,
                "confidence_score": f"Confidence: {int(conf * 100)}%",
                "quantifiable_impact_rm": impact_str
            }

        
        elif any(w in h for w in ["rent", "property", "real estate", "commercial", "premises", "sewa", "hartanah"]):
            rent_impact_pct = pct if pct > 0 else 0.05
            if tier == "survival":
                action = (
                    "Negotiate rent reduction or deferment with landlord immediately. "
                    "Consider downsizing premises or shifting to a home-based/co-working model."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a {rent_impact_pct:.0%} rent increase "
                    f"adds RM {round(cash_num * rent_impact_pct * 0.15):,}/month (assuming 15% rent-to-cost ratio). "
                    f"At survival-tier, premises cost is often the largest fixed expense — "
                    f"downsizing or renegotiating now is critical before the increase takes effect."
                ) if cash_num else (
                    f"Commercial rent increases of {rent_impact_pct:.0%} threaten fixed-cost stability. Negotiate immediately."
                )
                conf = 0.65
            elif tier == "stable":
                action = (
                    "Lock in a 2–3 year lease at current rates before increases take effect. "
                    "If expanding, negotiate tenant improvement allowances with the landlord."
                )
                explanation = (
                    f"With RM {cash_num:,} in cash, a {rent_impact_pct:.0%} rent increase "
                    f"(~RM {round(cash_num * rent_impact_pct * 0.15):,}/month) is manageable but erodes margins. "
                    f"Locking in a longer lease now protects against future increases "
                    f"and provides cost certainty for planning."
                ) if cash_num else "Commercial rent increases erode margins. Negotiate longer leases at current rates."
                conf = 0.62
            elif tier == "growth":
                action = (
                    "Consider purchasing commercial property instead of renting — current conditions "
                    "may favor buyers. Alternatively, negotiate revenue-sharing leases with landlords."
                )
                explanation = (
                    f"With RM {cash_num:,} in reserves, owning premises may be more cost-effective than renting. "
                    f"A {rent_impact_pct:.0%} rent increase over 5 years means RM {round(cash_num * rent_impact_pct * 0.15 * 60):,} "
                    f"in cumulative additional rent. Property ownership converts this expense into an asset."
                ) if cash_num else "Commercial rent increases make property ownership increasingly attractive."
                conf = 0.58
            else:
                action = "Review your current lease terms and negotiate before rent increases take effect. Enter your cash balance for a tailored strategy."
                explanation = "Commercial rent changes affect fixed operating costs. Negotiate lease terms proactively."
                conf = 0.38

            if cash_num:
                impact_str = f"RM {round(cash_num * rent_impact_pct * 0.15):,}/month — {rent_impact_pct:.0%} rent increase on ~15% rent-to-cost ratio"
            else:
                impact_str = "Cannot calculate without cash data. Assume 5% rent increase"

            return {
                "action_recommendation": action,
                "clear_explanation": explanation,
                "confidence_score": f"Confidence: {int(conf * 100)}%",
                "quantifiable_impact_rm": impact_str
            }

        if tier == "survival":
            action = (
                "Freeze all discretionary spending immediately. Build a 90-day cash reserve before considering "
                "any new commitments. Review every vendor contract for force majeure or exit clauses."
            )
            explanation = (
                f"With RM {cash_num:,} in cash, even indirect market disruption could be critical. "
                f"A conservative 2% downside risk equals RM {round(cash_num * gen_pct):,} — money you cannot afford to lose. "
                f"Survival-tier SMEs must prioritize liquidity above all else until the impact pathway of this news becomes clear."
            )
            conf = 0.55
        elif tier == "stable":
            action = (
                "Review vendor and customer contracts for exposure to this type of market event. "
                "Delay new large commitments for 30 days while monitoring developments."
            )
            explanation = (
                f"With RM {cash_num:,} in reserves, a 2% risk exposure equals RM {round(cash_num * gen_pct):,} — "
                f"manageable but not trivial. Stable-tier SMEs should run a 30-day monitoring window "
                f"before committing capital, while reviewing contracts for protective clauses."
            )
            conf = 0.50
        elif tier == "growth":
            action = (
                "Assess whether this market event creates a strategic opportunity — "
                "acquire undervalued assets, negotiate better vendor terms, or expand market share "
                "while competitors pull back."
            )
            explanation = (
                f"With RM {cash_num:,} in reserves, a 2% exposure (RM {round(cash_num * gen_pct):,}) is absorbable. "
                f"Growth-tier SMEs can turn market uncertainty into advantage by moving while others hesitate. "
                f"The stronger balance sheet allows opportunistic positioning rather than defensive contraction."
            )
            conf = 0.52
        else:
            action = "Maintain a 90-day cash reserve and monitor for direct impact on your operations. Enter your cash balance for a tailored strategy."
            explanation = (
                "Without financial data, assume moderate downside risk. "
                "Monitor the situation and enter your Cash Balance and Revenue for a specific calculation."
            )
            conf = 0.30

        if cash_num:
            impact_str = f"RM {round(cash_num * gen_pct):,} — estimated 2% operational exposure on RM {cash_num:,} cash"
        else:
            impact_str = "Cannot calculate without cash data. Assume 2% downside risk"

        return {
            "action_recommendation": action,
            "clear_explanation": explanation,
            "confidence_score": f"Confidence: {int(conf * 100)}%",
            "quantifiable_impact_rm": impact_str
        }


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text_content = ""
    for page in doc:
        text_content += page.get_text()
    doc.close()
    return text_content.strip()


def _close_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing unclosed strings,
    arrays, and objects so that json.loads() can parse it.

    Handles the common case where the model ran out of tokens mid-output.
    """
    if not text:
        return text

    
    import re
    text = re.sub(r"```\s*(?:json|JSON)?\s*\n?", "", text).strip()

    # Find the JSON body
    first_brace = text.find("{")
    if first_brace == -1:
        return text
    json_fragment = text[first_brace:]

    depth = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0

    for i, ch in enumerate(json_fragment):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            if not in_string:
                last_valid_pos = i
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                
                return text
        elif ch in ",:\n":
            last_valid_pos = i

    
    result = json_fragment

   
    if in_string:
        result += '"'


    result = re.sub(r',\s*"\w+"\s*:\s*$', '', result)
    result = re.sub(r',\s*$', '', result)

    
    open_brackets = result.count("[") - result.count("]")
    for _ in range(max(0, open_brackets)):
        result += "]"

    
    open_braces = result.count("{") - result.count("}")
    for _ in range(max(0, open_braces)):
        result += "}"

    return result


def _robust_json_parse(raw_text: str) -> dict | None:
    """Extract and parse JSON from AI response that may contain conversational
    text, markdown fences, trailing commas, or other malformations.

    Strategy: strip fences → brace extraction (string-aware) → repair → multi-pass parse.
    """
    import re

    if not raw_text:
        return None

    text = raw_text.strip()

    
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    
    text = re.sub(r"```\s*(?:json|JSON)?\s*\n?", "", text)
    text = text.strip()

    
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    
    try:
        repaired = _close_truncated_json(text)
        if repaired != text:
            return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

   
    first_brace = text.find("{")
    if first_brace == -1:
        return None

   
    depth = 0
    in_string = False
    escape_next = False
    last_brace = -1
    for i in range(first_brace, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_brace = i
                break

    if last_brace == -1:
        return None

    json_str = text[first_brace:last_brace + 1]

    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    
    def remove_trailing_commas(s):
        result = []
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if esc:
                result.append(ch)
                esc = False
                continue
            if ch == "\\":
                result.append(ch)
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                result.append(ch)
                continue
            if in_str:
                result.append(ch)
                continue
            if ch == "," and i + 1 < len(s) and s[i + 1] in "}]":
                continue  
            result.append(ch)
        return "".join(result)

    repaired = remove_trailing_commas(json_str)

    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    
    def fix_quotes(s):
        result = []
        in_single = False
        in_double = False
        esc = False
        for ch in s:
            if esc:
                result.append(ch)
                esc = False
                continue
            if ch == "\\":
                result.append(ch)
                esc = True
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                result.append(ch)
                continue
            if ch == "'" and not in_double:
                
                in_single = not in_single
                result.append('"')
                continue
            result.append(ch)
        return "".join(result)

    fixed = fix_quotes(repaired)

    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    
    try:
        unquoted = re.sub(r'(?<!["\w])(\w+)\s*:', r'"\1":', fixed)
        return json.loads(unquoted)
    except (json.JSONDecodeError, ValueError):
        pass

    
    try:
        no_comments = re.sub(r"//.*?\n", "\n", fixed)
        no_comments = re.sub(r"/\*.*?\*/", "", no_comments, flags=re.DOTALL)
        return json.loads(no_comments)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = {}
        for m in re.finditer(r'"(\w+)"\s*:\s*("[^"]*"|\d+\.?\d*|null|true|false)', fixed):
            key, val = m.group(1), m.group(2)
            try:
                result[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                result[key] = val.strip('"')
        if result:
            return result
    except Exception:
        pass

    return None


def _scrape_key_value(raw_text: str, cash) -> dict | None:
    """Scrape key-value pairs directly from raw AI output using regex.
    Handles truncated JSON where json.loads fails but the raw string
    still contains the data in 'key': 'value' patterns.
    """
    import re

    if not raw_text:
        return None

    result = {}

    
    m = re.search(r'"vendor"\s*:\s*"([^"]*)"?', raw_text, re.I)
    if m:
        result["vendor"] = m.group(1).strip() or None

    
    m = re.search(r'"date"\s*:\s*"([^"]*)"?', raw_text, re.I)
    if m:
        result["date"] = m.group(1).strip() or None

    
    m = re.search(r'"total"\s*:\s*"?([\d,]+\.?\d*)"?', raw_text, re.I)
    if m:
        try:
            result["total"] = float(m.group(1).replace(",", ""))
        except ValueError:
            result["total"] = m.group(1)

    
    m = re.search(r'"currency"\s*:\s*"([^"]*)"?', raw_text, re.I)
    if m:
        result["currency"] = m.group(1).strip() or None

    
    if "vendor" not in result:
        m = re.search(r'"(?:supplier|company)"\s*:\s*"([^"]*)"?', raw_text, re.I)
        if m:
            result["vendor"] = m.group(1).strip()

    if "total" not in result:
        for alt in ["amount", "total_amount", "grand_total"]:
            m = re.search(rf'"{alt}"\s*:\s*"?([\d,]+\.?\d*)"?', raw_text, re.I)
            if m:
                try:
                    result["total"] = float(m.group(1).replace(",", ""))
                except ValueError:
                    result["total"] = m.group(1)
                break

    if not result:
        return None

    
    amount = result.get("total")
    vendor = result.get("vendor")
    date = result.get("date")
    currency = result.get("currency", "RM")

    
    strategic_insight = None
    if amount and cash and cash > 0:
        ratio = amount / cash
        if ratio <= 0.3:
            strategic_insight = f"{currency} {amount:,.2f} invoice is {ratio:.0%} of {currency} {cash:,.0f} cash — manageable."
        elif ratio <= 0.7:
            strategic_insight = f"{currency} {amount:,.2f} invoice consumes {ratio:.0%} of {currency} {cash:,.0f} cash — consider installments."
        else:
            strategic_insight = f"{currency} {amount:,.2f} invoice is {ratio:.0%} of {currency} {cash:,.0f} cash — high risk, delay payment."

    return {
        "vendor": vendor,
        "invoice_date": date,
        "due_date": None,
        "amount": amount,
        "tax": round(amount * 0.08, 2) if amount else None,
        "po_reference": None,
        "currency": currency,
        "confidence": {
            "vendor": 0.6 if vendor else 0.0,
            "invoice_date": 0.5 if date else 0.0,
            "amount": 0.7 if amount else 0.0,
            "currency": 0.6 if currency else 0.0,
        },
        "strategic_insight": strategic_insight
    }


def _local_minimal_extract(text: str) -> dict | None:
    """Bare-minimum local fallback: find 'Total' or 'RM' in raw text
    so the user at least sees a number even when all AI fails.
    """
    import re

    if not text:
        return None

    
    amount = None
    total_patterns = [
        r"(?:total|jumlah|grand\s*total)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:total|jumlah|grand\s*total)\s*[:\-]?\s*MYR\s*([\d,]+\.?\d*)",
        r"RM\s*([\d,]+\.?\d{2})",
    ]
    for pattern in total_patterns:
        matches = list(re.finditer(pattern, text, re.I))
        if matches:
            for m in reversed(matches):
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 0:
                        amount = val
                        break
                except ValueError:
                    continue
            if amount:
                break

    
    lines = text.split("\n")
    vendor = None
    for line in lines[:max(1, len(lines) // 5)]:
        stripped = line.strip()
        if stripped and len(stripped) > 3 and re.search(r"(?:Sdn\s*Bhd|Bhd|Enterprise|Trading|Corp|Co\.|PLT)", stripped, re.I):
            vendor = stripped
            break
    if not vendor:
        for line in lines[:max(1, len(lines) // 5)]:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                vendor = stripped
                break

    cash = st.session_state.get("cash_balance")
    strategic_insight = None
    if amount and cash and cash > 0:
        ratio = amount / cash
        if ratio <= 0.3:
            strategic_insight = f"RM {amount:,.2f} invoice is {ratio:.0%} of RM {cash:,.0f} cash — manageable."
        elif ratio <= 0.7:
            strategic_insight = f"RM {amount:,.2f} invoice consumes {ratio:.0%} of RM {cash:,.0f} cash — consider installments."
        else:
            strategic_insight = f"RM {amount:,.2f} invoice is {ratio:.0%} of RM {cash:,.0f} cash — high risk, delay payment."

    return {
        "vendor": vendor,
        "invoice_date": None,
        "due_date": None,
        "amount": amount,
        "tax": round(amount * 0.08, 2) if amount else None,
        "po_reference": None,
        "currency": "RM",
        "confidence": {
            "vendor": 0.5 if vendor else 0.0,
            "invoice_date": 0.0,
            "amount": 0.6 if amount else 0.0,
            "currency": 0.5,
        },
        "strategic_insight": strategic_insight
    }


def extract_invoice_with_ai(text_content: str) -> dict:
    """3-tier invoice extraction:

    1. Try Gemini → parse JSON, or scrape key-value pairs from raw output
    2. If Gemini fails, try Z.AI GLM → parse JSON, or scrape key-value pairs
    3. If both AI fail, fall back to minimal local regex extraction
    """
    if not text_content:
        return None

    cash = st.session_state.get("cash_balance")

    user_prompt = (
        f"Extract data from this invoice text. Return ONLY raw JSON with keys: "
        f"vendor, date, total, currency. No markdown, no conversational text. "
        f"Do not extract line items, descriptions, or terms. "
        f"Input Text: {text_content}"
    )

    def _map_to_display(parsed):
        """Map the slim 4-key AI output to the display schema the UI expects."""
        result = {}

        
        result["vendor"] = parsed.get("vendor")
        result["invoice_date"] = parsed.get("date") or parsed.get("invoice_date")
        result["due_date"] = parsed.get("due_date")
        result["po_reference"] = parsed.get("po_reference")

        
        amount = parsed.get("total") or parsed.get("amount")
        if isinstance(amount, str):
            try:
                amount = float(amount.replace(",", ""))
            except (ValueError, TypeError):
                amount = None
        result["amount"] = amount

        
        result["currency"] = parsed.get("currency", "RM")

        # May not use
        tax = parsed.get("tax")
        if tax is None and amount:
            tax = round(amount * 0.08, 2)
        result["tax"] = tax

        
        if isinstance(parsed.get("confidence"), dict):
            result["confidence"] = parsed["confidence"]
        else:
            result["confidence"] = {
                "vendor": 0.7 if result["vendor"] else 0.0,
                "invoice_date": 0.7 if result["invoice_date"] else 0.0,
                "amount": 0.8 if result["amount"] else 0.0,
                "currency": 0.7 if result.get("currency") else 0.0,
            }

        
        if isinstance(parsed.get("strategic_insight"), str) and parsed["strategic_insight"]:
            result["strategic_insight"] = parsed["strategic_insight"]
        else:
            result["strategic_insight"] = None
            if amount and cash and cash > 0:
                ratio = amount / cash
                cur = result["currency"]
                if ratio <= 0.3:
                    result["strategic_insight"] = f"{cur} {amount:,.2f} invoice is {ratio:.0%} of {cur} {cash:,.0f} cash — manageable."
                elif ratio <= 0.7:
                    result["strategic_insight"] = f"{cur} {amount:,.2f} invoice consumes {ratio:.0%} of {cur} {cash:,.0f} cash — consider installments."
                else:
                    result["strategic_insight"] = f"{cur} {amount:,.2f} invoice is {ratio:.0%} of {cur} {cash:,.0f} cash — high risk, delay payment."

        return result

    
    gemini_key = st.session_state.get("gemini_key", "")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key, client_options={"api_endpoint": "generativelanguage.googleapis.com"})
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction="You extract invoice data into JSON. Return ONLY raw JSON. No markdown fences, no explanations."
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    stop_sequences=["}\n"]
                )
            )
            raw = response.text
            if raw:
                parsed = _robust_json_parse(raw)
                if parsed is not None:
                    return _map_to_display(parsed)

                # JSON parse failed — try key-value regex scraping
                scraped = _scrape_key_value(raw, cash)
                if scraped and scraped.get("amount"):
                    return scraped

                st.warning("Could not extract structured data from Gemini response. Trying Z.AI...")
                with st.expander("Debug: Raw Gemini Response"):
                    st.text(raw)
            else:
                st.warning("Gemini returned an empty response. Trying Z.AI...")
        except Exception as e:
            st.warning(f"Gemini API call failed ({e}). Trying Z.AI...")

    
    zai_key = st.session_state.get("zai_key", "")
    if zai_key:
        try:
            payload = {
                "model": "ilmu-glm-5.1",
                "messages": [
                    {"role": "system", "content": "You extract invoice data into JSON. Return ONLY raw JSON. No markdown fences, no explanations."},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 256
            }
            headers = {
                "Authorization": f"Bearer {zai_key}",
                "Content-Type": "application/json"
            }
            resp = requests.post(ZAI_API_URL, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            if raw:
                parsed = _robust_json_parse(raw)
                if parsed is not None:
                    return _map_to_display(parsed)

                # JSON parse failed — try key-value regex scraping
                scraped = _scrape_key_value(raw, cash)
                if scraped and scraped.get("amount"):
                    return scraped

                st.warning("Could not extract structured data from Z.AI response. Falling back to local extraction.")
                with st.expander("Debug: Raw Z.AI Response"):
                    st.text(raw)
            else:
                st.warning("Z.AI GLM returned an empty response. Falling back to local extraction.")
        except Exception as e:
            st.warning(f"Z.AI GLM API call failed ({e}). Falling back to local extraction.")

    
    return _local_minimal_extract(text_content)


def _scrape_ai_text(ai_text: str, cash) -> dict:
    """Scrape financial values from the AI's raw text response.

    The AI already read and understood the invoice — even if it didn't
    return valid JSON, its text contains the extracted values. This function
    uses targeted regex to pull them out.

    Returns a dict with the same structure as the JSON output, with
    confidence scores reflecting that this was scraped, not structured.
    """
    import re

    if not ai_text:
        return None

    
    amount = None
    amount_patterns = [
        r"(?:total\s*amount|amount|grand\s*total|invoice\s*amount|total\s*due)\s*(?:is|:)?\s*RM\s*([\d,]+\.?\d*)",
        r"RM\s*([\d,]+\.?\d{2})\b",  # Any RM value with 2 decimal places
    ]
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, ai_text, re.I))
        if matches:
            # Take the last (usually the grand total) or the largest
            for m in reversed(matches):
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 0:
                        amount = val
                        break
                except ValueError:
                    continue
            if amount:
                break

    
    vendor = None
    vendor_patterns = [
        r"(?:vendor|supplier|company|from|seller)\s*(?:is|:)?\s*([A-Z][\w\s&]*(?:Sdn\s*Bhd|Bhd|Enterprise|Trading|Corp|PLT|Co\.))",
        r"(?:vendor|supplier|company)\s*(?:is|:)?\s*(.+?)(?:\n|\.|,)",
    ]
    for pattern in vendor_patterns:
        m = re.search(pattern, ai_text, re.I)
        if m:
            vendor = m.group(1).strip()
            break

    
    tax = None
    tax_patterns = [
        r"(?:SST|GST|sales\s*tax|service\s*tax|tax)\s*(?:is|:|amount)?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:SST|GST)\s*(?:at|of)?\s*(\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in tax_patterns:
        m = re.search(pattern, ai_text, re.I)
        if m:
            val = m.group(1).replace(",", "")
            try:
                tax = float(val)
                break
            except ValueError:
                # Might be a percentage — calculate from amount
                if amount and "%" in m.group(0):
                    try:
                        pct = float(val) / 100
                        tax = round(amount * pct, 2)
                        break
                    except ValueError:
                        pass

    # Also here tax may not being used
    if tax is None and amount:
        tax = round(amount * 0.08, 2)

    
    invoice_date = None
    m = re.search(r"(?:invoice\s*date|date\s*of\s*issue)\s*(?:is|:)?\s*(\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4})", ai_text, re.I)
    if m:
        invoice_date = m.group(1).strip()

    # May not be important in this system for now
    due_date = None
    m = re.search(r"(?:due\s*date|payment\s*due|pay\s*by)\s*(?:is|:)?\s*(\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4})", ai_text, re.I)
    if m:
        due_date = m.group(1).strip()

    
    po_reference = None
    m = re.search(r"(?:PO|purchase\s*order|order\s*ref)\s*(?:is|:|number|no)?\s*[:\-]?\s*([\w\-]+)", ai_text, re.I)
    if m:
        po_reference = m.group(1).strip()

   
    strategic_insight = None
    insight_patterns = [
        r"(?:strategic\s*insight|financial\s*impact|risk\s*assessment|this\s*invoice)\s*[:\-]?\s*(.+?)(?:\n\n|$)",
        r"(?:represents|equivalent\s*to|consumes|is\s*\d+%?\s*of)\s*.+?(?:\n\n|$)",
    ]
    for pattern in insight_patterns:
        m = re.search(pattern, ai_text, re.I | re.DOTALL)
        if m:
            strategic_insight = m.group(1).strip()[:300]  # cap length
            break

    
    if not strategic_insight and amount and cash and cash > 0:
        ratio = amount / cash
        if ratio <= 0.3:
            strategic_insight = f"RM {amount:,.2f} invoice is {ratio:.0%} of RM {cash:,.0f} cash — manageable."
        elif ratio <= 0.7:
            strategic_insight = f"RM {amount:,.2f} invoice consumes {ratio:.0%} of RM {cash:,.0f} cash — consider installments."
        else:
            strategic_insight = f"RM {amount:,.2f} invoice is {ratio:.0%} of RM {cash:,.0f} cash — high risk, delay payment."

    
    confidence = {
        "vendor": 0.6 if vendor else 0.0,
        "invoice_date": 0.55 if invoice_date else 0.0,
        "due_date": 0.55 if due_date else 0.0,
        "amount": 0.7 if amount else 0.0,
        "tax": 0.4 if tax and tax > 0 else 0.0,
        "po_reference": 0.5 if po_reference else 0.0,
    }

    return {
        "vendor": vendor or "Extracted from AI text",
        "invoice_date": invoice_date,
        "due_date": due_date,
        "amount": amount,
        "tax": tax,
        "po_reference": po_reference,
        "confidence": confidence,
        "strategic_insight": strategic_insight
    }


def _local_extract(text: str) -> dict:
    """Advanced regex fallback when no AI API is available.
    Uses multiple keyword variations for each field, layout-aware heuristics,
    Malaysian-specific patterns, per-field confidence scoring, and a
    strategic_insight generated from the extracted data vs. cash balance.
    """
    import re

    lines = text.split("\n")

    def _find(patterns, flags=re.I):
        """Try multiple patterns in order, return first match."""
        if isinstance(patterns, str):
            patterns = [patterns]
        for p in patterns:
            m = re.search(p, text, flags)
            if m:
                return m.group(1).strip()
        return None

    def _find_amount(patterns, flags=re.I):
        """Try multiple patterns for monetary amounts."""
        if isinstance(patterns, str):
            patterns = [patterns]
        for p in patterns:
            m = re.search(p, text, flags)
            if m:
                val = m.group(1).replace(",", "").strip()
                try:
                    return float(val)
                except ValueError:
                    continue
        return None

    def _find_last_amount(patterns, flags=re.I):
        """Find the LAST match for an amount — typically the grand total at bottom."""
        if isinstance(patterns, str):
            patterns = [patterns]
        last_val = None
        for p in patterns:
            for m in re.finditer(p, text, flags):
                val = m.group(1).replace(",", "").strip()
                try:
                    last_val = float(val)
                except ValueError:
                    continue
        return last_val

    # Top 20% of lines
    top_lines = lines[: max(1, len(lines) // 5)]

    vendor = _find([
        r"(?:vendor|supplier|from|seller|company|issued\s*by|billed\s*from)\s*[:\-]?\s*(.+)",
        r"(?:nama|syarikat)\s*[:\-]?\s*(.+)",
    ])
    if not vendor:
        for line in top_lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3 and not re.match(r"^(invoice|tax|date|no|ref|page)", stripped, re.I):
                if re.search(r"(?:Sdn\s*Bhd|Bhd|Sendirian|Enterprise|Trading|Corp|Co\.|PLT)", stripped, re.I):
                    vendor = stripped
                    break
    if not vendor:
        for line in top_lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                vendor = stripped
                break

    
    invoice_date = _find([
        r"(?:invoice\s*date|date\s*of\s*issue|tarikh\s*invoin|date)\s*[:\-]?\s*(\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4})",
    ])

    
    due_date = _find([
        r"(?:due\s*date|payment\s*due|pay\s*by|tarikh\s*bayaran|bayaran\s*sebelum)\s*[:\-]?\s*(\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4})",
    ])

    
    amount_patterns = [
        r"(?:grand\s*total|jumlah\s*besar)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:total\s*(?:amount|due|payable)|jumlah\s*(?:keseluruhan|bayaran))\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:balance\s*due|amount\s*due|baki\s*perlu)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:total|jumlah)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"RM\s*([\d,]+\.?\d{2})\s*$",
    ]
    amount = _find_last_amount(amount_patterns)

    
    if amount is None:
        bottom_lines = lines[-max(1, len(lines) // 5):]
        bottom_text = "\n".join(bottom_lines)
        last_rm = list(re.finditer(r"RM\s*([\d,]+\.?\d{2})", bottom_text, re.I))
        if last_rm:
            try:
                amount = float(last_rm[-1].group(1).replace(",", ""))
            except ValueError:
                pass

    
    if amount is None:
        all_numbers = re.findall(r"([\d,]+\.\d{2})", text)
        if all_numbers:
            try:
                amount = max(float(n.replace(",", "")) for n in all_numbers)
            except ValueError:
                pass

    
    tax = _find_amount([
        r"(?:SST|sales\s*and\s*services?\s*tax|cukai\s*penjualan)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:GST|goods\s*and\s*services?\s*tax)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:service\s*tax|cukai\s*perkhidmatan)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:tax|cukai)\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
        r"(?:tax|gst|sst)\s*[\(%]\s*\d*\.?\d*\s*[%)]\s*[:\-]?\s*RM\s*([\d,]+\.?\d*)",
    ])

    
    if tax is None and amount is not None and amount > 0:
        tax = round(amount * 0.08, 2)

    # May not be used in this system
    po_ref = _find([
        r"(?:po\s*reference|purchase\s*order|po\s*no|no\s*po|rujukan\s*po)\s*[:\-]?\s*(.+)",
        r"(?:order\s*ref|reference\s*no|no\s*rujukan)\s*[:\-]?\s*(.+)",
    ])

    
    confidence = {
        "vendor": 0.9 if vendor and re.search(r"(?:Sdn\s*Bhd|Bhd|Enterprise|Corp|PLT)", vendor, re.I) else (0.5 if vendor else 0.0),
        "invoice_date": 0.85 if invoice_date else 0.0,
        "due_date": 0.85 if due_date else 0.0,
        "amount": 0.9 if amount and amount > 0 else 0.0,
        "tax": 0.85 if tax and tax > 0 and _find_amount([r"(?:SST|GST|tax|cukai)\s*[:\-]?\s*RM\s*"]) is not None else (0.4 if tax else 0.0),
        "po_reference": 0.8 if po_ref else 0.0,
    }

    
    cash = st.session_state.get("cash_balance")
    if amount and cash and cash > 0:
        ratio = amount / cash
        remaining = cash - amount
        if ratio <= 0.3:
            insight = (
                f"This RM {amount:,.2f} invoice is {ratio:.0%} of the RM {cash:,.0f} cash balance. "
                f"After payment, RM {remaining:,.0f} remains — manageable with sufficient working capital."
            )
        elif ratio <= 0.7:
            insight = (
                f"This RM {amount:,.2f} invoice consumes {ratio:.0%} of the RM {cash:,.0f} cash balance, "
                f"leaving only RM {remaining:,.0f}. Consider negotiating installment terms to preserve liquidity."
            )
        else:
            insight = (
                f"This RM {amount:,.2f} invoice represents {ratio:.0%} of the RM {cash:,.0f} cash balance — "
                f"payment would critically deplete or exceed available funds. Delay and seek extended terms."
            )
    elif amount:
        insight = f"Cannot assess risk — no cash balance on record. Invoice amount: RM {amount:,.2f}. Enter your Cash Balance in the Cash Flow page."
    else:
        insight = "Could not extract invoice amount. Upload a clearer PDF or enter details manually."

    return {
        "vendor": vendor,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "amount": amount,
        "tax": tax,
        "po_reference": po_ref,
        "confidence": confidence,
        "strategic_insight": insight
    }


def compute_payment_decision(extracted: dict, cash_balance: float) -> dict:
    """Compare invoice amount against cash balance and produce a dynamic recommendation.

    Returns JSON-serializable dict with: action, reasoning_path, risk_level.
    """
    amount = extracted.get("amount")
    if amount is None:
        try:
            amount = float(extracted.get("amount", 0))
        except (ValueError, TypeError):
            amount = 0

    if cash_balance is None or cash_balance <= 0:
        return {
            "action": "Delay",
            "reasoning_path": [
                f"Unstructured data (invoice): Amount = RM {amount:,.2f}",
                "Structured data (bank): No cash balance on record — user has not entered data in Cash Flow page",
                "Decision: Cannot confirm funds; default to Delay to avoid overdraft risk"
            ],
            "risk_level": "high"
        }

    ratio = amount / cash_balance

    if ratio <= 0.3:
        action = "Pay Now"
        risk = "low"
        reason = (
            f"Invoice amount (RM {amount:,.2f}) is {ratio:.0%} of cash balance (RM {cash_balance:,.0f}). "
            f"Remaining after payment: RM {cash_balance - amount:,.0f}. Sufficient liquidity to pay immediately."
        )
    elif ratio <= 0.7:
        action = "Negotiate Installments"
        risk = "medium"
        reason = (
            f"Invoice amount (RM {amount:,.2f}) is {ratio:.0%} of cash balance (RM {cash_balance:,.0f}). "
            f"Paying in full would leave only RM {cash_balance - amount:,.0f}, which may strain operations. "
            f"Negotiate installment terms to preserve working capital."
        )
    else:
        action = "Delay"
        risk = "high"
        reason = (
            f"Invoice amount (RM {amount:,.2f}) is {ratio:.0%} of cash balance (RM {cash_balance:,.0f}). "
            f"Paying would exceed or critically deplete available funds. "
            f"Delay payment and seek extended terms or emergency funding."
        )

    reasoning_path = [
        f"Unstructured data (invoice): Vendor = {extracted.get('vendor', 'N/A')}, "
        f"Amount = RM {amount:,.2f}, Due = {extracted.get('due_date', 'N/A')}",
        f"Structured data (bank): Cash balance = RM {cash_balance:,.0f}",
        f"Ratio analysis: Invoice is {ratio:.0%} of available cash",
        f"Decision: {action} — {reason}"
    ]

    return {
        "action": action,
        "reasoning_path": reasoning_path,
        "risk_level": risk
    }


if "ai_suggestion" not in st.session_state:
    st.session_state.ai_suggestion = None
if "ai_reasoning" not in st.session_state:
    st.session_state.ai_reasoning = None
if "cash_balance" not in st.session_state:
    st.session_state.cash_balance = None
if "monthly_revenue" not in st.session_state:
    st.session_state.monthly_revenue = None
if "doc_scan_result" not in st.session_state:
    st.session_state.doc_scan_result = None
if "monthly_burn" not in st.session_state:
    st.session_state.monthly_burn = None
if "runway_days" not in st.session_state:
    st.session_state.runway_days = None
if "net_burn" not in st.session_state:
    st.session_state.net_burn = None
if "reasoning_steps" not in st.session_state:
    st.session_state.reasoning_steps = None
if "reasoning_source" not in st.session_state:
    st.session_state.reasoning_source = None
if "reasoning_confidence" not in st.session_state:
    st.session_state.reasoning_confidence = None
if "reasoning_impact" not in st.session_state:
    st.session_state.reasoning_impact = None


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.title("SME Finance AI")
    page = st.radio("Navigate", [
        "Dashboard Overview",
        "Cash Flow Runway",
        "AI Insight Generator",
        "Document Scanner",
        "Decision Explainability"
    ], index=0)

    st.divider()

    st.subheader("Decision Explainability")
    if st.session_state.ai_suggestion:
        st.success(f"Suggestion: {st.session_state.ai_suggestion}")
        with st.expander("Why did the AI suggest this?"):
            st.write(st.session_state.ai_reasoning or "No reasoning available yet.")
    else:
        st.info("Run the AI Insight Generator first to see explainability here.")

    st.divider()

    
    with st.expander("API Keys", expanded=not st.session_state.gemini_key and not st.session_state.zai_key):
        st.caption("Gemini is tried first, then Z.AI GLM as backup. Without either key, the dashboard uses a local calculation engine.")
        gemini_input = st.text_input("Gemini API Key", value=st.session_state.gemini_key, type="password", key="gemini_key_input")
        zai_input = st.text_input("Z.AI GLM API Key", value=st.session_state.zai_key, type="password", key="zai_key_input")
        if st.button("Save Keys"):
            st.session_state.gemini_key = gemini_input
            st.session_state.zai_key = zai_input
            st.success("API keys saved for this session.")
        gemini_status = "Connected" if st.session_state.gemini_key else "No key"
        zai_status = "Connected" if st.session_state.zai_key else "No key"
        st.info(f"Gemini: {gemini_status}  |  Z.AI GLM: {zai_status}")


if page == "Dashboard Overview":
    st.title("SME Financial Health Dashboard")
    st.caption("UM Hackathon 2026 Real-time financial intelligence for SMEs")

    
    cash = st.session_state.cash_balance
    revenue = st.session_state.monthly_revenue
    burn = st.session_state.monthly_burn
    net = st.session_state.net_burn
    runway = st.session_state.runway_days

    has_data = cash is not None

    
    if has_data:
        net_cash = revenue - (burn or 0)
        runway_display = f"{runway:.0f} days" if runway and runway < 999 else "Cash-flow positive"

        
        if runway and runway < 30:
            runway_delta = "Critical"
            runway_color = "#ef4444"
        elif runway and runway < 90:
            runway_delta = "Caution"
            runway_color = "#f59e0b"
        else:
            runway_delta = "Healthy"
            runway_color = "#22c55e"

        net_sign = "▲" if net_cash >= 0 else "▼"
        net_color = "#22c55e" if net_cash >= 0 else "#ef4444"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Monthly Revenue</div><div class="metric-value">RM {revenue:,.0f}</div><div class="metric-delta" style="color:#22c55e">From Cash Flow page</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Operating Costs</div><div class="metric-value">RM {burn:,.0f}</div><div class="metric-delta" style="color:#ef4444">From Cash Flow page</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Net Cash Flow</div><div class="metric-value">RM {net_cash:,.0f}</div><div class="metric-delta" style="color:{net_color}">{net_sign} {"surplus" if net_cash >= 0 else "deficit"}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Runway Left</div><div class="metric-value">{runway_display}</div><div class="metric-delta" style="color:{runway_color}">{runway_delta}</div></div>', unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-label">Monthly Revenue</div><div class="metric-value">—</div><div class="metric-delta" style="color:#94a3b8">No data yet</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-label">Operating Costs</div><div class="metric-value">—</div><div class="metric-delta" style="color:#94a3b8">No data yet</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-label">Net Cash Flow</div><div class="metric-value">—</div><div class="metric-delta" style="color:#94a3b8">No data yet</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-label">Runway Left</div><div class="metric-value">—</div><div class="metric-delta" style="color:#94a3b8">No data yet</div></div>', unsafe_allow_html=True)

    
    st.subheader("Quantifiable Impact")
    imp1, imp2 = st.columns(2)
    if has_data and net is not None:
        with imp1:
            st.metric("Cash Balance", f"RM {cash:,.0f}", f"{'Surplus' if net < 0 else 'Deficit'} of RM {abs(net):,.0f}/month")
        with imp2:
            savings_estimate = round(abs(net) * 0.15) if net else 0
            st.metric("AI Cost Savings Opportunity", f"RM {savings_estimate:,.0f}", "15% of net burn via optimization")
    else:
        with imp1:
            st.metric("Cash Balance", "—", "Enter data in Cash Flow page")
        with imp2:
            st.metric("AI Cost Savings Opportunity", "—", "Enter data in Cash Flow page")

    
    st.divider()
    if has_data:
        st.subheader("Live Data from Cash Flow Runway")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Cash Balance", f"RM {cash:,.0f}")
        with summary_col2:
            st.metric("Monthly Net Burn", f"RM {net:,.0f}" if net else "N/A")
        with summary_col3:
            daily = net / 30 if net else 0
            st.metric("Daily Burn Rate", f"RM {daily:,.0f}" if net else "N/A")
        st.success("Metrics are live-linked. Update them in the Cash Flow Runway page.")
    else:
        st.info("No data yet. Go to **Cash Flow Runway** to enter your numbers — they'll appear here automatically.")


elif page == "Cash Flow Runway":
    st.title("Cash Flow Runway Predictor")
    st.caption("Estimate how many days your business can operate before cash runs out.")

    
    default_cash = st.session_state.cash_balance if st.session_state.cash_balance else 50000
    default_burn = st.session_state.monthly_burn if st.session_state.monthly_burn else 30000
    default_rev = st.session_state.monthly_revenue if st.session_state.monthly_revenue else 20000

    with st.form("runway_form"):
        st.subheader("Input Parameters")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            cash_balance = st.number_input("Current Cash Balance (RM)", min_value=0, value=int(default_cash), step=1000)
        with col_b:
            monthly_burn = st.number_input("Monthly Operating Expenses (RM)", min_value=0, value=int(default_burn), step=500)
        with col_c:
            monthly_revenue = st.number_input("Monthly Revenue (RM)", min_value=0, value=int(default_rev), step=500)

        seasonal_adj = st.slider("Seasonal Adjustment Factor", 0.5, 2.0, 1.0, 0.1,
                                  help="Adjust for seasonal revenue dips (e.g., 0.5 = half revenue expected)")
        submitted = st.form_submit_button("Calculate Runway", type="primary")

    if submitted:
        adjusted_revenue = monthly_revenue * seasonal_adj
        net_burn = monthly_burn - adjusted_revenue

        if net_burn <= 0:
            st.success(f"Your business is cash-flow positive! Net monthly surplus: RM {abs(net_burn):,.0f}")
            runway_days = 999  # effectively infinite
        else:
            runway_days = (cash_balance / net_burn) * 30
            st.markdown(f"### Runway Estimate: **{runway_days:.0f} days**")

            if runway_days < 30:
                st.error("Critical: Less than 1 month of runway remaining!")
            elif runway_days < 90:
                st.warning("Caution: Less than 3 months of runway. Consider cost-cutting or funding.")
            else:
                st.success("Healthy: Over 3 months of runway available.")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Daily Net Burn", f"RM {net_burn/30:,.0f}")
            with col_m2:
                st.metric("Monthly Net Burn", f"RM {net_burn:,.0f}")
            with col_m3:
                st.metric("Runway End Date", f"{datetime.now().strftime('%b %Y')} + {runway_days:.0f}d")

        
        st.session_state.cash_balance = cash_balance
        st.session_state.monthly_revenue = monthly_revenue
        st.session_state.monthly_burn = monthly_burn
        st.session_state.net_burn = net_burn
        st.session_state.runway_days = runway_days

        st.divider()
        if st.button("Save to Dashboard", type="primary", key="save_to_dash"):
            st.success("Saved! Switch to Dashboard Overview to see your live metrics.")

    
    if st.session_state.runway_days is not None:
        st.divider()
        st.caption("Current saved data (visible on Dashboard Overview):")
        saved_col1, saved_col2, saved_col3 = st.columns(3)
        with saved_col1:
            st.metric("Saved Cash Balance", f"RM {st.session_state.cash_balance:,.0f}" if st.session_state.cash_balance else "N/A")
        with saved_col2:
            st.metric("Saved Monthly Revenue", f"RM {st.session_state.monthly_revenue:,.0f}" if st.session_state.monthly_revenue else "N/A")
        with saved_col3:
            st.metric("Saved Runway", f"{st.session_state.runway_days:.0f} days" if st.session_state.runway_days else "N/A")


elif page == "AI Insight Generator":
    st.title("AI Insight Generator")
    st.caption("Decision Intelligence Engine — CFO-level analysis with localized Malaysian context.")

    headline = st.text_input("Enter a news headline or event", placeholder="e.g., 'Ringgit drops 3% against USD amid global uncertainty'")

    if st.button("Generate Insight", type="primary") and headline:
        with st.spinner("Decision Intelligence Engine is analyzing..."):
            result = generate_ai_insight(headline)

        if "error" in result:
            st.error(f"AI error: {result['error']}")
        else:
            action = result.get("action_recommendation", "N/A")
            explanation = result.get("clear_explanation", "N/A")
            confidence = result.get("confidence_score", "Confidence: 0%")
            impact = result.get("quantifiable_impact_rm", "N/A")

            # Parse confidence percentage for the progress bar
            import re
            conf_match = re.search(r"(\d+)", confidence)
            conf_num = int(conf_match.group(1)) / 100 if conf_match else 0

            st.markdown("### Action Recommendation")
            st.markdown(f'<div class="insight-box">{action}</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Clear Explanation")
                st.write(explanation)
            with col_b:
                st.subheader("Quantifiable Impact (RM)")
                st.markdown(f'<p style="font-size:0.85rem; color:#f1f5f9;">{impact}</p>', unsafe_allow_html=True)

            # Dynamic Risk Assessment — Confidence Score
            st.subheader("Dynamic Risk Assessment")
            st.progress(conf_num, text=confidence)

            
            st.session_state.ai_suggestion = action
            st.session_state.ai_reasoning = (
                f"Headline: '{headline}'\n\n"
                f"Action: {action}\n\n"
                f"Explanation: {explanation}\n\n"
                f"Confidence: {confidence}\n\n"
                f"Impact: {impact}"
            )
            cash_val = st.session_state.get("cash_balance")
            rev_val = st.session_state.get("monthly_revenue")
            st.session_state.reasoning_source = "ai_insight"
            st.session_state.reasoning_confidence = confidence
            st.session_state.reasoning_impact = impact
            st.session_state.reasoning_steps = [
                f"**Input Parsing** — Analyzed the headline: '{headline}' for financial keywords, sentiment, and Malaysian economic relevance.",
                f"**Localized Context** — Mapped the headline to BNM policy, OPR trends, SST regulations, and Ringgit stability where applicable.",
                f"**Cash-Position Correlation** — Cross-referenced the headline against Cash Balance: {'RM ' + f'{cash_val:,.0f}' if cash_val else 'unknown'} and Revenue: {'RM ' + f'{rev_val:,.0f}' if rev_val else 'unknown'} to determine business-specific exposure.",
                f"**Recommendation Generation** — Produced a non-linear action tailored to the business's liquidity tier (survival / stable / growth), not a generic response.",
                f"**Impact Quantification** — {impact}"
            ]

            
            with st.expander("View Raw JSON"):
                st.json(result)

            st.info("Check the sidebar or the Decision Explainability tab for the full reasoning.")


elif page == "Document Scanner":
    st.title("Document Scanner")
    st.caption("Upload a PDF invoice (in RM only) — AI extracts key data and recommends a payment action based on your cash position.")

    if st.session_state.cash_balance is not None:
        st.info(f"Using cash balance from Cash Flow page: **RM {st.session_state.cash_balance:,.0f}**")
    else:
        st.warning("No cash balance on record. Enter your balance in the Cash Flow page first for accurate recommendations.")

    uploaded_file = st.file_uploader("Upload Invoice PDF", type=["pdf"], help="Supports single-page and multi-page PDFs")

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        with st.spinner("Extracting text from PDF..."):
            text_content = extract_text_from_pdf(uploaded_file)

        if not text_content:
            st.error("Could not extract any text from this PDF. It may be a scanned image — OCR support coming soon.")
        else:
            with st.expander("View Extracted Raw Text"):
                st.text(text_content)

            with st.spinner("AI is parsing invoice fields (Gemini → Z.AI → local)..."):
                extracted = extract_invoice_with_ai(text_content)

            if extracted is None:
                st.error("AI extraction returned no results. Please check the PDF content.")
            else:
                # ── Extracted Fields ──
                st.subheader("Extracted Invoice Data")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Key Details**")
                    cur = extracted.get("currency", "RM")
                    fields = {
                        "Vendor": extracted.get("vendor"),
                        "Invoice Date": extracted.get("invoice_date"),
                        f"Amount ({cur})": extracted.get("amount"),
                    }
                    for key, value in fields.items():
                        display = value if value is not None else "Not found"
                        st.write(f"- **{key}:** {display}")

                with col2:
                    st.markdown("**Extraction Confidence**")
                    conf = extracted.get("confidence") or {}
                    field_key_map = {
                        "Vendor": "vendor",
                        "Invoice Date": "invoice_date",
                        "Amount": "amount",
                        "Currency": "currency",
                    }
                    for display_name, key in field_key_map.items():
                        score = conf.get(key, 0)
                        if score:
                            st.progress(score, text=f"{display_name}: {score:.0%}")
                        else:
                            st.progress(0.0, text=f"{display_name}: not found")

                
                insight = extracted.get("strategic_insight")
                if insight:
                    st.divider()
                    st.subheader("Strategic Insight")
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

                
                source = "Gemini" if st.session_state.get("gemini_key") else ("Z.AI GLM" if st.session_state.get("zai_key") else "Local regex fallback")
                st.caption(f"Extraction method: {source} | Text length: {len(text_content)} chars")

                
                st.divider()

                amount_val = extracted.get("amount")
                if amount_val is not None:
                    try:
                        amount_val = float(amount_val)
                    except (ValueError, TypeError):
                        amount_val = None

                if amount_val is not None and amount_val > 0:
                    decision = compute_payment_decision(extracted, st.session_state.cash_balance)

                    st.subheader("Payment Recommendation")

                    action_colors = {"Pay Now": "🟢", "Negotiate Installments": "🟡", "Delay": "🔴"}
                    action_icon = action_colors.get(decision["action"], "⚪")

                    st.markdown(f"### {action_icon} {decision['action']}")
                    st.markdown(f"**Risk Level:** {decision['risk_level'].upper()}")

                    
                    imp1, imp2, imp3 = st.columns(3)
                    with imp1:
                        st.metric("Invoice Amount", f"RM {amount_val:,.2f}")
                    with imp2:
                        balance = st.session_state.cash_balance or 0
                        st.metric("Cash Balance", f"RM {balance:,.0f}")
                    with imp3:
                        remaining = balance - amount_val
                        st.metric("After Payment", f"RM {remaining:,.0f}",
                                  delta=f"{remaining:,.0f}", delta_color="inverse" if remaining < 0 else "off")

                    
                    st.subheader("Reasoning Path")
                    st.caption("How the AI combined unstructured (invoice) and structured (bank balance) data to reach its decision.")

                    for i, step in enumerate(decision["reasoning_path"], 1):
                        st.markdown(f"**Step {i}:** {step}")

                    
                    balance = st.session_state.cash_balance or 0
                    ratio = amount_val / balance if balance > 0 else 0
                    st.session_state.ai_suggestion = f"{decision['action']} — Invoice of RM {amount_val:,.2f}"
                    st.session_state.ai_reasoning = " → ".join(decision["reasoning_path"])
                    st.session_state.reasoning_source = "document_scanner"
                    st.session_state.reasoning_confidence = f"Confidence: {int(min(0.9, max(0.4, 1.0 - ratio)) * 100)}%"
                    st.session_state.reasoning_impact = f"RM {amount_val:,.2f} outflow represents {ratio:.0%} of RM {balance:,.0f} current liquidity"
                    st.session_state.reasoning_steps = [
                        f"**Document Parsing** — Extracted unstructured text from the uploaded PDF invoice.",
                        f"**AI Field Extraction** — Identified Vendor: {extracted.get('vendor', 'N/A')}, Amount: RM {amount_val:,.2f}, Due: {extracted.get('due_date', 'N/A')} using AI (Gemini → Z.AI → local).",
                        f"**Cash Position Lookup** — Retrieved current Cash Balance of RM {balance:,.0f} from the Cash Flow Runway page.",
                        f"**Ratio Analysis** — Calculated that an RM {amount_val:,.2f} outflow represents {ratio:.0%} of your current liquidity (RM {balance:,.0f}).",
                        f"**Decision Logic** — At {ratio:.0%} exposure, the recommendation is '{decision['action']}' (Risk: {decision['risk_level'].upper()}). "
                        f"After payment, RM {balance - amount_val:,.0f} would remain."
                    ]

                    
                    with st.expander("View Raw JSON Output"):
                        output_json = {
                            "extracted_fields": extracted,
                            "payment_decision": decision
                        }
                        st.json(output_json)
                else:
                    st.error("Could not determine invoice amount from the extracted data. The PDF may not contain a recognizable total.")


elif page == "Decision Explainability":
    st.title("Decision Explainability Panel")
    st.caption("Understand the reasoning behind every AI-generated financial recommendation.")

    if st.session_state.ai_suggestion:
        st.subheader("Current AI Suggestion")
        st.success(st.session_state.ai_suggestion)

        
        source = st.session_state.get("reasoning_source", "unknown")
        if source == "ai_insight":
            st.caption("Source: AI Insight Generator")
        elif source == "document_scanner":
            st.caption("Source: Document Scanner")
        else:
            st.caption("Source: Previous session")

        
        st.subheader("Reasoning Chain")
        steps = st.session_state.get("reasoning_steps")
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}:** {step}")
        else:
            # Fallback: show the raw reasoning text
            st.info(st.session_state.ai_reasoning or "No reasoning data available.")

        
        st.subheader("Confidence & Impact")
        col_c1, col_c2 = st.columns(2)

        conf = st.session_state.get("reasoning_confidence", "")
        with col_c1:
            if conf:
                # Parse percentage for progress bar
                import re
                conf_match = re.search(r"(\d+)", str(conf))
                conf_num = int(conf_match.group(1)) / 100 if conf_match else 0.5
                st.progress(conf_num, text=str(conf))
            else:
                st.metric("Model Confidence", "—")

        impact = st.session_state.get("reasoning_impact", "")
        with col_c2:
            if impact:
                st.metric("Quantified Impact", impact)
            else:
                st.metric("Quantified Impact", "—")

        
        st.subheader("Full Reasoning")
        st.info(st.session_state.ai_reasoning)
    else:
        st.info("No AI suggestion generated yet. Go to the AI Insight Generator or Document Scanner tab first.")