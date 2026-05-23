"""
NBFC Collection Allocation Engine — v2.0
==========================================
Key design corrections from v1:

  1. Payment mode is a DISTRIBUTION across history, not a single fixed value.
     Each borrower has a % breakdown of how they paid (UPI, eNACH, BBPS, QR, Cash)
     across all recorded EMIs. A single customer may have paid 40% UPI, 30% BBPS,
     20% QR and 10% Cash — this is far more realistic.

  2. PTP data NOT available. Telecalling scoring is rebuilt using only:
       - Call pick-up rate
       - Number of connected conversations
       - Call response consistency
       - Switchoff frequency
       - Guarantor reachability

  3. Channel priority is CHEAPEST FIRST:
       Priority 1: AI Calling   (cheapest, scalable)
       Priority 2: Telecalling  (moderate cost, human)
       Priority 3: Field Visit  (expensive, last resort)

     Field visit is only assigned when:
       (a) Contactability is critically low AND no response to calls, OR
       (b) Historical data shows customer paid ONLY after physical visits, OR
       (c) Cash-dominant payer with high delay (physical collection needed)

  4. The goal is to MAXIMIZE AI + Tele allocations and minimize field visits.

Usage:
  pip install pandas tabulate
  python collection_allocation_engine_v2.py
"""

import pandas as pd
from tabulate import tabulate

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DUMMY DATA
# ─────────────────────────────────────────────────────────────────────────────
# Payment mode is now a DISTRIBUTION across EMI payment history.
# Columns: upi_pct, enach_pct, bbps_pct, qr_pct, cash_pct  → must sum to 100
#
# Contactability fields (NO PTP — not available):
#   call_pickup_rate      → % of calls answered out of total dialled
#   connected_calls       → absolute count of calls where conversation happened
#   total_calls_dialled   → total calls made to borrower
#   switchoff_freq        → how many times phone was switched off / unreachable
#   guarantor_reachable   → 1 if guarantor was ever successfully contacted, 0 if not
#   number_active         → 1 if number is currently active, 0 if inactive/wrong
#
# Behavioral fields:
#   avg_delay_days        → average EMI delay across repayment history
#   max_delay_days        → worst single delay episode
#   on_time_pct           → % of EMIs paid on or before due date
#   bounce_count          → total EMI bounces (NACH return / cheque bounce)
#   missed_emi_count      → EMIs completely missed (unpaid as of today)
#   field_visits_done     → number of field visits already attempted
#   paid_after_field      → number of times payment came AFTER a field visit
#   paid_after_call       → number of times payment came after any call (AI or human)
#   outstanding_amount    → current overdue outstanding (INR)

RAW_DATA = [
    # ( loan_id, name,
    #   upi%, enach%, bbps%, qr%, cash%,
    #   pickup_rate, connected_calls, total_dialled, switchoff_freq,
    #   guarantor_reachable, number_active,
    #   avg_delay, max_delay, on_time_pct, bounce_count, missed_emi,
    #   field_visits_done, paid_after_field, paid_after_call,
    #   outstanding )

    # ── Highly Digital, Low Delay → Strong AI candidates ─────────────────────
    ("LN-0012","Rajesh Kumar",        60,30, 8, 2, 0,  88,12,14, 0, 1,1,  2, 5,94, 0,0, 0,0,10,  4200),
    ("LN-0047","Priya Sharma",        0,90, 8, 2, 0,  95,15,16, 0, 1,1,  1, 3,97, 0,0, 0,0,12,  3100),
    ("LN-0083","Amit Singh",          30,20,40, 8, 2,  76,10,13, 1, 1,1,  4, 9,85, 1,0, 0,0, 8,  8600),
    ("LN-0115","Sunita Patel",        55,25,15, 4, 1,  82,11,14, 0, 1,1,  3, 7,90, 0,0, 0,0, 9,  5500),
    ("LN-0129","Vikram Nair",         20,10,60, 8, 2,  70, 9,13, 1, 1,1,  5,10,78, 1,0, 0,0, 7,  9200),
    ("LN-0144","Kavitha Rao",         10,75,12, 2, 1,  90,14,16, 0, 1,1,  2, 6,92, 0,0, 0,0,11,  4800),
    ("LN-0189","Meena Joshi",         5,85, 8, 1, 1,  93,15,16, 0, 1,1,  1, 4,96, 0,0, 0,0,13,  2900),
    ("LN-0698","Renu Sharma",         65,20,10, 4, 1,  87,12,14, 0, 1,1,  3, 6,91, 0,0, 0,0,10,  5100),
    ("LN-0572","Neha Chandra",        70,15,10, 4, 1,  80,11,14, 0, 1,1,  4, 8,83, 1,0, 0,0, 8,  7400),

    # ── Mixed mode payers, moderate delay → Good Tele / borderline AI ────────
    ("LN-0156","Deepak Mehta",        15,10,45,20,10,  65, 8,13, 1, 1,1,  6,12,75, 2,1, 1,0, 6, 14200),
    ("LN-0168","Ananya Iyer",         40,10,25,15,10,  79,10,13, 0, 1,1,  3, 8,88, 1,0, 0,0, 7,  9800),
    ("LN-0172","Suresh Gupta",        10, 5,40,25,20,  60, 7,12, 2, 1,1,  7,14,72, 2,1, 1,0, 5, 17600),
    ("LN-0307","Pooja Singh",         20,10,40,20,10,  58, 7,12, 2, 1,1,  7,13,73, 2,1, 0,0, 5, 13400),
    ("LN-0558","Sanjay Patil",        25,10,40,15,10,  66, 8,13, 1, 1,1,  6,11,77, 1,0, 1,0, 6, 12800),
    ("LN-0617","Vivek Agarwal",       30,10,40,12, 8,  72, 9,13, 1, 1,1,  5,10,80, 1,0, 0,0, 7, 10600),
    ("LN-0628","Archana Pillai",      20,10,45,15,10,  63, 8,13, 1, 1,1,  7,12,74, 2,1, 0,0, 6, 13900),
    ("LN-0589","Ajay Kumar",          15, 5,45,20,15,  55, 6,11, 2, 1,1,  8,15,70, 2,1, 1,0, 4, 18200),
    ("LN-0671","Biswas Roy",          20,10,40,15,15,  52, 6,11, 2, 0,1, 10,18,67, 2,1, 1,0, 4, 20100),

    # ── Moderate-high delay, QR/BBPS mix, partial contactability → Tele ──────
    ("LN-0203","Ramesh Tiwari",        5, 0,20,55,20,  45, 4,10, 3, 0,1, 12,22,60, 3,1, 2,1, 3, 28500),
    ("LN-0218","Lalita Yadav",        10, 5,35,30,20,  50, 5,10, 2, 1,1, 10,19,65, 2,1, 1,0, 4, 22300),
    ("LN-0235","Prakash Verma",        5, 0,15,50,30,  40, 3, 9, 3, 0,1, 15,25,55, 3,2, 2,1, 2, 34700),
    ("LN-0247","Geeta Mishra",        10, 0,35,30,25,  55, 5,10, 2, 1,1,  9,17,68, 2,1, 1,0, 4, 19800),
    ("LN-0261","Naresh Pandey",        5, 0,15,45,35,  42, 4, 9, 3, 0,1, 11,21,62, 3,2, 2,1, 2, 30200),
    ("LN-0279","Sarita Chandra",      10, 5,30,30,25,  52, 5,10, 2, 1,1,  8,16,70, 2,1, 1,0, 4, 18900),
    ("LN-0292","Anil Kumar",           5, 0,10,40,45,  38, 3, 9, 4, 0,1, 13,24,58, 3,2, 2,1, 2, 32400),
    ("LN-0334","Rekha Pillai",        10, 5,30,30,25,  48, 4,10, 3, 0,1,  9,18,66, 2,1, 1,0, 3, 21500),
    ("LN-0478","Mahesh Gupta",         5, 0,20,45,30,  35, 3, 9, 4, 0,1, 16,28,52, 3,2, 2,1, 2, 37800),
    ("LN-0504","Pankaj Mishra",        5, 0,20,45,30,  38, 3, 9, 3, 0,1, 17,29,50, 3,2, 2,1, 2, 39200),
    ("LN-0603","Sangeeta Singh",       5, 0,15,45,35,  42, 3, 9, 3, 0,1, 11,20,63, 3,2, 2,1, 2, 29800),
    ("LN-0642","Sunil Patil",          5, 0,15,45,35,  36, 3, 9, 4, 0,1, 13,23,58, 3,2, 2,1, 2, 33100),
    ("LN-0712","Kishore Gupta",        5, 0,20,45,30,  46, 4,10, 2, 1,1,  9,17,68, 2,1, 1,0, 3, 20900),

    # ── Cash-dominant, still has some contactability → Tele before Field ──────
    ("LN-0319","Harish Agarwal",       2, 0, 5,20,73,  35, 2, 8, 4, 0,1, 14,26,56, 4,2, 3,2, 1, 41200),
    ("LN-0443","Leela Nair",           0, 0, 5,20,75,  25, 2, 8, 5, 0,1, 19,32,44, 4,2, 3,2, 1, 48700),
    ("LN-0491","Seema Joshi",          2, 0, 3,15,80,  20, 1, 7, 5, 0,1, 21,35,42, 5,3, 3,2, 1, 52300),

    # ── Proven field payers — pays ONLY after visit (critical, assign field) ──
    ("LN-0348","Ganesh Patil",         0, 0, 2, 8,90,  20, 1, 7, 5, 0,1, 22,38,40, 5,3, 4,3, 0, 58400),
    ("LN-0377","Mohan Lal",            0, 0, 3,10,87,  22, 1, 7, 5, 0,1, 18,33,45, 4,2, 5,4, 0, 49200),
    ("LN-0405","Ratan Singh",          0, 0, 2, 8,90,  18, 1, 6, 6, 0,1, 20,36,42, 5,3, 3,2, 0, 54600),

    # ── Completely unreachable / inactive number → direct field ───────────────
    ("LN-0362","Shanti Devi",          5, 0, 5,20,70,  15, 0, 6, 6, 0,0, 25,42,35, 5,3, 3,2, 0, 67300),
    ("LN-0391","Champa Bai",           0, 0, 2, 8,90,  10, 0, 5, 7, 0,0, 30,50,30, 6,4, 4,3, 0, 78900),
    ("LN-0418","Savita Kumari",        0, 0, 2,10,88,  12, 0, 5, 7, 0,0, 28,46,32, 6,3, 5,4, 0, 71200),
    ("LN-0429","Dinesh Sharma",        0, 0, 1, 5,94,   8, 0, 4, 8, 0,0, 35,58,25, 7,4, 6,5, 0, 92100),
    ("LN-0457","Brijesh Yadav",        0, 0, 1, 5,94,  10, 0, 5, 8, 0,0, 32,54,28, 6,4, 5,4, 0, 85400),
    ("LN-0462","Usha Tiwari",          0, 0, 2, 8,90,  14, 0, 5, 6, 0,0, 24,40,38, 5,3, 4,3, 0, 63800),
    ("LN-0518","Jyoti Verma",          0, 0, 2, 8,90,  15, 0, 5, 7, 0,0, 26,44,36, 5,3, 4,3, 0, 69500),
    ("LN-0531","Rohit Pandey",         0, 0, 1, 5,94,   12, 0, 4, 7, 0,0, 29,48,30, 6,4, 5,4, 0, 80200),
    ("LN-0547","Kamala Rao",           0, 0, 1, 4,95,   9, 0, 4, 9, 0,0, 33,55,26, 7,4, 6,5, 0, 94700),
    ("LN-0656","Madhuri Devi",         0, 0, 2, 8,90,  13, 0, 5, 7, 0,0, 27,45,34, 5,3, 4,3, 0, 72600),
    ("LN-0684","Tara Singh",           0, 0, 1, 3,96,   6, 0, 4,10, 0,0, 38,62,20, 8,5, 7,5, 0,105800),
]

COLUMNS = [
    "loan_id","name",
    "upi_pct","enach_pct","bbps_pct","qr_pct","cash_pct",
    "call_pickup_rate","connected_calls","total_dialled","switchoff_freq",
    "guarantor_reachable","number_active",
    "avg_delay_days","max_delay_days","on_time_pct","bounce_count","missed_emi_count",
    "field_visits_done","paid_after_field","paid_after_call",
    "outstanding_amount"
]

df = pd.DataFrame(RAW_DATA, columns=COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Digital Affinity Score from payment MODE DISTRIBUTION ────────────────
#
# Instead of a single mode, we weight the borrower's payment history:
#   UPI Autopay   → 100 pts  (fully digital, automated)
#   eNACH         →  90 pts  (automated debit)
#   BBPS          →  70 pts  (digital but manual initiation)
#   QR            →  35 pts  (semi-digital, often needs agent nudge)
#   Cash          →   5 pts  (physical dependency)
#
# digital_affinity = weighted average across all EMIs paid

DIGITAL_WEIGHTS = {
    "upi_pct":   1.00,
    "enach_pct": 0.90,
    "bbps_pct":  0.70,
    "qr_pct":    0.35,
    "cash_pct":  0.05,
}

def compute_digital_affinity(row) -> int:
    score = (
        row["upi_pct"]   * DIGITAL_WEIGHTS["upi_pct"]  +
        row["enach_pct"] * DIGITAL_WEIGHTS["enach_pct"] +
        row["bbps_pct"]  * DIGITAL_WEIGHTS["bbps_pct"]  +
        row["qr_pct"]    * DIGITAL_WEIGHTS["qr_pct"]    +
        row["cash_pct"]  * DIGITAL_WEIGHTS["cash_pct"]
    )
    # score is already on 0-100 scale (weighted % of 100)
    return round(score)

df["digital_affinity"] = df.apply(compute_digital_affinity, axis=1)

# Dominant payment mode (for reporting/human-readable label)
def dominant_mode(row) -> str:
    modes = {
        "UPI":   row["upi_pct"],
        "eNACH": row["enach_pct"],
        "BBPS":  row["bbps_pct"],
        "QR":    row["qr_pct"],
        "Cash":  row["cash_pct"],
    }
    return max(modes, key=modes.get)

df["dominant_mode"] = df.apply(dominant_mode, axis=1)
df["cash_dependency_pct"] = df["cash_pct"] + df["qr_pct"]   # physical payment %


# ── 2b. Contactability Score — NO PTP DATA ───────────────────────────────────
#
# Variables used (PTP excluded entirely):
#   call_pickup_rate     → 35% weight  (most direct reachability signal)
#   conversation_rate    → 25% weight  (connected calls / total dialled)
#   number_active        → 20% weight  (binary — is number alive?)
#   switchoff_penalty    → 15% deduction (avoidance signal)
#   guarantor_reachable  →  5% weight  (backup contact)
#
# This gives a 0–100 score purely on reachability.

def compute_contactability(row) -> int:
    conversation_rate = (
        row["connected_calls"] / max(row["total_dialled"], 1)
    ) * 100

    pickup_w       = row["call_pickup_rate"]  * 0.35
    conv_w         = conversation_rate         * 0.25
    active_w       = row["number_active"]     * 20     # 0 or 20
    switchoff_pen  = min(row["switchoff_freq"] * 4, 20)  # max 20 penalty
    guar_w         = row["guarantor_reachable"] * 5    # 0 or 5

    raw = pickup_w + conv_w + active_w + guar_w - switchoff_pen
    return max(0, min(100, round(raw)))

df["contactability_score"] = df.apply(compute_contactability, axis=1)


# ── 2c. Call Response Consistency ────────────────────────────────────────────
# How consistently does the borrower answer when we call?
# pickup_rate alone can be misleading — consistency matters too.
# We penalise high switchoff even if pickup is moderate.

def call_consistency(row) -> str:
    rate = row["call_pickup_rate"]
    soff = row["switchoff_freq"]
    if rate >= 70 and soff <= 1:  return "High"
    if rate >= 45 and soff <= 3:  return "Medium"
    if rate >= 20 and soff <= 5:  return "Low"
    return "Critical"

df["call_consistency"] = df.apply(call_consistency, axis=1)


# ── 2d. Physical Collection Dependency (Field Affinity) ──────────────────────
# Does this customer ONLY pay when someone shows up physically?
# Signal: multiple field visits done AND payments consistently came AFTER visits.

def field_affinity_flag(row) -> bool:
    if row["field_visits_done"] < 2:
        return False
    visit_pay_rate = row["paid_after_field"] / max(row["field_visits_done"], 1)
    call_pay_rate  = row["paid_after_call"]  / max(row["total_dialled"], 1)
    # Field affinity = visits converted well AND calls rarely converted
    return visit_pay_rate >= 0.60 and call_pay_rate < 0.15

df["field_affinity"] = df.apply(field_affinity_flag, axis=1)


# ── 2e. Delay Severity ───────────────────────────────────────────────────────
def delay_band(row) -> str:
    d = row["avg_delay_days"]
    if d <= 5:   return "Green"    # low risk
    if d <= 12:  return "Yellow"   # moderate
    if d <= 22:  return "Orange"   # high
    return "Red"                   # critical

df["delay_band"] = df.apply(delay_band, axis=1)
df["delay_score"] = (100 - df["avg_delay_days"] * 2.0).clip(lower=0).round().astype(int)


# ── 2f. Composite Risk Score ─────────────────────────────────────────────────
# Higher score = lower risk = better candidate for cheaper channels
df["composite_score"] = (
    df["contactability_score"] * 0.40 +
    df["digital_affinity"]     * 0.25 +
    df["delay_score"]          * 0.20 +
    df["on_time_pct"]          * 0.15
).round().astype(int)


# ── 2g. Customer Persona ─────────────────────────────────────────────────────
def assign_persona(row) -> str:
    cs  = row["contactability_score"]
    da  = row["digital_affinity"]
    delay = row["avg_delay_days"]
    cash  = row["cash_dependency_pct"]

    if da >= 70 and delay <= 5:
        return "Digital Disciplined"
    if cs < 20 and delay > 20:
        return "Chronic Avoider"
    if cash >= 80 and delay > 15:
        return "Cash Dependent"
    if row["field_affinity"]:
        return "Pressure Responder"
    if delay > 10 and cs > 40:
        return "Soft Delinquent"
    if row["guarantor_reachable"] == 0 and cs < 35:
        return "Untraceable"
    return "Moderate Risk"

df["persona"] = df.apply(assign_persona, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ALLOCATION ENGINE (Cheapest-First Logic)
# ─────────────────────────────────────────────────────────────────────────────
#
# The OBJECTIVE is: maximise AI + Tele allocation, minimise field visits.
# Field visit is ONLY assigned under specific hard conditions — not default fallback.
#
# Decision tree (evaluated top to bottom, first match wins):
#
#  STEP 1 — Hard Field Triggers (bypass cheap channels entirely)
#    → Number inactive AND no guarantor AND high delay      → Field Visit
#    → Proven field payer (field_affinity=True)             → Field Visit
#    → Cash-dominant (≥75%) AND delay ≥ 18 days            → Field Visit
#    → Contactability critically low (< 15) AND delay > 20 → Field Visit
#
#  STEP 2 — AI Calling (cheapest, try first)
#    → Contactability ≥ 55 AND digital_affinity ≥ 50 AND delay ≤ 8
#    → Contactability ≥ 65 AND delay ≤ 12 (even if mixed mode)
#
#  STEP 3 — Telecalling (human, moderate cost — DEFAULT fallback)
#    → Everyone else who doesn't hit field triggers
#    → This is the WIDEST bucket intentionally
#
#  STEP 4 — Field Visit (only if nothing else applies)
#    → Remaining cases with critical contactability or physical dependency

def allocate_channel(row) -> tuple:
    """
    Returns (channel, reason) tuple.
    """
    cs      = row["contactability_score"]
    da      = row["digital_affinity"]
    delay   = row["avg_delay_days"]
    cash    = row["cash_dependency_pct"]
    active  = row["number_active"]
    guar    = row["guarantor_reachable"]
    fa      = row["field_affinity"]
    paid_af = row["paid_after_field"]
    visits  = row["field_visits_done"]
    paid_ac = row["paid_after_call"]

    # ── STEP 1: Hard Field Triggers ───────────────────────────────────────────
    if active == 0 and guar == 0 and delay > 15:
        return ("Field Visit", "Number inactive, no guarantor, high delay")

    if fa:
        return ("Field Visit", f"Proven field payer: {paid_af}/{visits} visits converted, calls rarely work")

    if cash >= 75 and delay >= 18:
        return ("Field Visit", f"Cash-dominant ({cash}% physical) with {delay}d delay — physical collection needed")

    if cs < 15 and delay > 20:
        return ("Field Visit", f"Critically unreachable (CS={cs}) with severe delay ({delay}d)")

    # ── STEP 2: AI Calling ────────────────────────────────────────────────────
    if cs >= 65 and da >= 50 and delay <= 8:
        return ("AI Calling", f"High contactability ({cs}), digital-leaning ({da}% affinity), low delay ({delay}d)")

    if cs >= 55 and da >= 65 and delay <= 12:
        return ("AI Calling", f"Strong digital affinity ({da}%), good pickup ({cs}), moderate delay")

    if cs >= 70 and delay <= 10:
        return ("AI Calling", f"Very high contactability ({cs}) — digital mode secondary")

    # ── STEP 3: Telecalling (intentionally wide bucket) ──────────────────────
    # Covers: moderate contactability, mixed mode payers, moderate delay,
    # any case that doesn't need field and isn't a clean AI candidate
    if cs >= 30:
        return ("Telecalling", f"Contactable ({cs}) — human agent can negotiate, field not yet justified")

    if cs >= 15 and delay <= 25:
        return ("Telecalling", f"Low-moderate contactability ({cs}) but delay manageable ({delay}d) — attempt tele before field")

    # ── STEP 4: Field Visit (true last resort) ────────────────────────────────
    return ("Field Visit", f"Contactability too low ({cs}) for remote channels — physical intervention required")


allocation_results = df.apply(allocate_channel, axis=1)
df["allocated_channel"] = allocation_results.apply(lambda x: x[0])
df["allocation_reason"] = allocation_results.apply(lambda x: x[1])


# ── Priority ─────────────────────────────────────────────────────────────────
def assign_priority(row) -> str:
    if row["avg_delay_days"] > 20 or row["contactability_score"] < 15:
        return "High"
    if row["avg_delay_days"] > 10 or row["contactability_score"] < 40:
        return "Medium"
    return "Low"

df["priority"] = df.apply(assign_priority, axis=1)


# ── Escalation Path ───────────────────────────────────────────────────────────
def escalation_path(row) -> str:
    ch = row["allocated_channel"]
    if ch == "AI Calling":
        return "AI → Telecalling → Field"
    if ch == "Telecalling":
        return "Tele → Field"
    return "Field (terminal)"

df["escalation_path"] = df.apply(escalation_path, axis=1)


# ── Success Probability Estimates ────────────────────────────────────────────
# Heuristic model — not ML, but grounded in the features we have.
def channel_probs(row):
    cs    = row["contactability_score"]
    da    = row["digital_affinity"]
    delay = row["avg_delay_days"]
    cash  = row["cash_dependency_pct"]

    ai_p    = int(min(92, cs * 0.50 + da * 0.30 + max(0, 15 - delay) * 1.0))
    tele_p  = int(min(88, cs * 0.45 + 20 + max(0, 20 - delay) * 0.8))
    field_p = int(min(85, 45 + cash * 0.25 + (100 - cs) * 0.20 + min(delay * 0.4, 15)))
    return ai_p, tele_p, field_p

probs = df.apply(channel_probs, axis=1)
df["ai_success_prob"]    = probs.apply(lambda x: x[0])
df["tele_success_prob"]  = probs.apply(lambda x: x[1])
df["field_success_prob"] = probs.apply(lambda x: x[2])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "═" * 90)
    print(f"  {title}")
    print("═" * 90)


def report_summary():
    section("ALLOCATION SUMMARY — CHANNEL SPLIT")
    counts = df["allocated_channel"].value_counts()
    total  = len(df)
    print(f"\n  {'Channel':<18} {'Cases':>6}  {'Share':>7}  {'Visual'}")
    print(f"  {'─'*18} {'─'*6}  {'─'*7}  {'─'*30}")
    for ch, cnt in counts.items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {ch:<18} {cnt:>6}  {pct:>6.1f}%  {bar}")
    print(f"\n  {'Total':<18} {total:>6}")


def report_priority_matrix():
    section("PRIORITY × CHANNEL MATRIX")
    pivot = df.groupby(["allocated_channel","priority"]).size().unstack(fill_value=0)
    print(tabulate(pivot, headers="keys", tablefmt="rounded_outline"))


def report_persona():
    section("PERSONA DISTRIBUTION")
    persona_ch = df.groupby(["persona","allocated_channel"]).size().unstack(fill_value=0)
    print(tabulate(persona_ch, headers="keys", tablefmt="rounded_outline"))


def report_full_roster():
    section("FULL ALLOCATION ROSTER")
    display = df[[
        "loan_id","name","avg_delay_days","dominant_mode","digital_affinity",
        "contactability_score","composite_score","persona",
        "allocated_channel","priority","escalation_path","allocation_reason"
    ]].copy()
    display.columns = [
        "Loan ID","Name","Delay(d)","Dom. Mode","Digital%",
        "Contact","Composite","Persona",
        "Channel","Priority","Escalation","Reason"
    ]
    print(tabulate(display, headers="keys", tablefmt="rounded_outline",
                   showindex=False, maxcolwidths=28))


def report_high_priority():
    section("HIGH PRIORITY CASES — IMMEDIATE ACTION")
    high = df[df["priority"] == "High"][[
        "loan_id","name","avg_delay_days","contactability_score",
        "cash_dependency_pct","outstanding_amount",
        "allocated_channel","allocation_reason"
    ]]
    print(tabulate(high, headers=[
        "Loan ID","Name","Delay","Contact","Cash+QR%","Outstanding(₹)","Channel","Reason"
    ], tablefmt="rounded_outline", showindex=False))


def report_field_cases():
    section("FIELD VISIT CASES — WHY FIELD WAS CHOSEN")
    field_df = df[df["allocated_channel"] == "Field Visit"][[
        "loan_id","name","avg_delay_days","contactability_score",
        "cash_dependency_pct","field_affinity","number_active",
        "outstanding_amount","allocation_reason"
    ]]
    print(tabulate(field_df, headers=[
        "Loan ID","Name","Delay","Contact","Cash+QR%","Field Affinity",
        "Num Active","Outstanding(₹)","Reason"
    ], tablefmt="rounded_outline", showindex=False))


def report_avg_by_channel():
    section("AVERAGE SCORES BY CHANNEL")
    agg = df.groupby("allocated_channel").agg(
        Count            = ("loan_id",              "count"),
        Avg_Contact      = ("contactability_score", "mean"),
        Avg_Digital      = ("digital_affinity",     "mean"),
        Avg_Delay        = ("avg_delay_days",        "mean"),
        Avg_Composite    = ("composite_score",       "mean"),
        Avg_Outstanding  = ("outstanding_amount",    "mean"),
        Avg_AI_Prob      = ("ai_success_prob",       "mean"),
        Avg_Tele_Prob    = ("tele_success_prob",     "mean"),
        Avg_Field_Prob   = ("field_success_prob",    "mean"),
    ).round(1)
    print(tabulate(agg, headers="keys", tablefmt="rounded_outline"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SINGLE BORROWER DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

def deep_dive(loan_id: str):
    row = df[df["loan_id"] == loan_id]
    if row.empty:
        print(f"  Loan {loan_id} not found.")
        return
    r = row.iloc[0]
    section(f"CASE DEEP DIVE — {r['loan_id']} / {r['name']}")

    print(f"\n  ── PAYMENT MODE DISTRIBUTION ──────────────────────────────")
    modes = [("UPI Autopay", r["upi_pct"]), ("eNACH", r["enach_pct"]),
             ("BBPS", r["bbps_pct"]), ("QR Code", r["qr_pct"]), ("Cash", r["cash_pct"])]
    for mode, pct in modes:
        bar = "▓" * int(pct / 3)
        print(f"  {mode:<14} {pct:>3}%  {bar}")
    print(f"  → Dominant mode: {r['dominant_mode']}  |  Digital affinity: {r['digital_affinity']}%")

    print(f"\n  ── CONTACTABILITY (NO PTP) ────────────────────────────────")
    print(f"  Call pick-up rate      : {r['call_pickup_rate']}%")
    print(f"  Connected calls        : {r['connected_calls']} of {r['total_dialled']} dialled")
    print(f"  Switchoff frequency    : {r['switchoff_freq']}")
    print(f"  Number active          : {'Yes' if r['number_active'] else 'No'}")
    print(f"  Guarantor reachable    : {'Yes' if r['guarantor_reachable'] else 'No'}")
    print(f"  → Contactability score : {r['contactability_score']}")
    print(f"  → Call consistency     : {r['call_consistency']}")

    print(f"\n  ── REPAYMENT BEHAVIOUR ────────────────────────────────────")
    print(f"  Avg delay              : {r['avg_delay_days']} days")
    print(f"  Max delay              : {r['max_delay_days']} days")
    print(f"  On-time payment %      : {r['on_time_pct']}%")
    print(f"  Bounce count           : {r['bounce_count']}")
    print(f"  Missed EMIs            : {r['missed_emi_count']}")
    print(f"  Delay band             : {r['delay_band']}")

    print(f"\n  ── CHANNEL HISTORY ────────────────────────────────────────")
    print(f"  Field visits done      : {r['field_visits_done']}")
    print(f"  Paid after field visit : {r['paid_after_field']}")
    print(f"  Paid after any call    : {r['paid_after_call']}")
    print(f"  Field affinity flag    : {'YES' if r['field_affinity'] else 'No'}")

    print(f"\n  ── SCORES & DECISION ──────────────────────────────────────")
    print(f"  Digital affinity       : {r['digital_affinity']}")
    print(f"  Contactability score   : {r['contactability_score']}")
    print(f"  Composite score        : {r['composite_score']}")
    print(f"  Outstanding (₹)        : ₹{r['outstanding_amount']:,.0f}")
    print(f"  Persona                : {r['persona']}")
    print(f"  Priority               : {r['priority']}")
    print(f"\n  ★ ALLOCATED CHANNEL    : {r['allocated_channel']}")
    print(f"  ★ REASON               : {r['allocation_reason']}")
    print(f"  ★ ESCALATION PATH      : {r['escalation_path']}")
    print(f"\n  Channel success probabilities:")
    print(f"    AI Calling    : {r['ai_success_prob']}%")
    print(f"    Telecalling   : {r['tele_success_prob']}%")
    print(f"    Field Visit   : {r['field_success_prob']}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ESCALATION PLAN PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def escalation_plan(loan_id: str):
    row = df[df["loan_id"] == loan_id]
    if row.empty:
        return
    r = row.iloc[0]
    ch = r["allocated_channel"]
    section(f"ESCALATION PLAN — {loan_id} ({r['name']})")
    print(f"  Starting channel: {ch}")

    if ch == "AI Calling":
        stages = [
            ("1","AI Calling",
             "Contactability high, digital affinity sufficient",
             "Max 3 automated IVR + WhatsApp attempts over 48 hrs",
             "Payment received OR clear commitment given",
             "No pickup / call dropped / no action after 3 attempts"),
            ("2","Telecalling",
             "AI unresolved — human agent takes over",
             "5 calls over 5 days. Document all dispositions.",
             "Payment received OR restructuring agreed",
             "No response / avoidance / no commitment"),
            ("3","Field Visit",
             "Remote channels exhausted",
             "2 visits. Verify address, collect or legal notice.",
             "Payment collected",
             "—")
        ]
    elif ch == "Telecalling":
        stages = [
            ("1","Telecalling",
             "Moderate contactability — human agent first",
             "5 calls over 5 days",
             "Payment received",
             "No response or continued avoidance"),
            ("2","Field Visit",
             "Tele unresolved",
             "2 visits, physical collection or legal notice",
             "Payment collected",
             "—")
        ]
    else:
        stages = [
            ("1","Field Visit",
             r["allocation_reason"],
             "Up to 3 visits. If no resolution → legal.",
             "Payment collected",
             "Legal escalation if 3 visits fail")
        ]

    for s in stages:
        num, ch_name, trigger, action, success, fail = s
        print(f"\n  ┌─ Stage {num}: {ch_name}")
        print(f"  │  Trigger  : {trigger}")
        print(f"  │  Action   : {action}")
        print(f"  │  Success  : {success}")
        if fail != "—":
            print(f"  └─ Fail → {fail}")
        else:
            print(f"  └─ Terminal stage")
        if stages.index(s) < len(stages) - 1:
            print(f"            ↓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(path: str = "allocation_output_v2.csv"):
    out_cols = [
        "loan_id","name",
        "upi_pct","enach_pct","bbps_pct","qr_pct","cash_pct",
        "dominant_mode","digital_affinity","cash_dependency_pct",
        "call_pickup_rate","connected_calls","total_dialled","switchoff_freq",
        "number_active","guarantor_reachable",
        "contactability_score","call_consistency",
        "avg_delay_days","max_delay_days","on_time_pct","bounce_count","missed_emi_count",
        "delay_band","delay_score",
        "field_visits_done","paid_after_field","paid_after_call","field_affinity",
        "outstanding_amount",
        "digital_affinity","composite_score","persona",
        "allocated_channel","priority","escalation_path","allocation_reason",
        "ai_success_prob","tele_success_prob","field_success_prob"
    ]
    # deduplicate cols
    seen = set()
    out_cols = [c for c in out_cols if not (c in seen or seen.add(c))]
    df[out_cols].to_csv(path, index=False)
    print(f"\n  ✓ Exported to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "▓" * 90)
    print("  NBFC COLLECTION ALLOCATION ENGINE  v2.0")
    print("  Cheapest-channel-first routing  |  48 Borrowers  |  May 2026")
    print("  Key changes: mode distribution, no PTP, maximise AI+Tele")
    print("▓" * 90)

    report_summary()
    report_priority_matrix()
    report_persona()
    report_avg_by_channel()
    report_full_roster()
    report_high_priority()
    report_field_cases()

    # Individual deep dives
    deep_dive("LN-0012")   # Digital disciplined → AI
    deep_dive("LN-0203")   # Mixed QR/Cash, moderate delay → Tele
    deep_dive("LN-0348")   # Proven field payer → Field
    deep_dive("LN-0391")   # Inactive number, no guarantor → Field

    # Escalation plans
    escalation_plan("LN-0083")   # AI case
    escalation_plan("LN-0203")   # Tele case
    escalation_plan("LN-0348")   # Field case

    export_csv()

    print("\n" + "▓" * 90)
    print("  ENGINE RUN COMPLETE")
    print("▓" * 90 + "\n")
