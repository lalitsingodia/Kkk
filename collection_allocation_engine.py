"""
NBFC Collection Allocation Engine
==================================
Intelligent channel routing for collections:
  - AI Calling
  - Telecalling
  - Field Visit

Architecture:
  1. Feature Engineering  (contactability, delay, mode, channel affinity)
  2. Scoring Engine       (weighted composite scores)
  3. Decision Engine      (rule-based allocation with escalation path)
  4. Reporting            (allocation summary + persona analytics)

Usage:
  pip install pandas tabulate
  python collection_allocation_engine.py
"""

import pandas as pd
from tabulate import tabulate

# ─────────────────────────────────────────────────────────────────────────────
# 1. DUMMY DATA — 48 Borrowers
# ─────────────────────────────────────────────────────────────────────────────

RAW_DATA = [
    # (loan_id, name, avg_delay_days, on_time_pct, bounce_freq, missed_emi,
    #  payment_mode, call_pickup_rate, ptp_honored_pct, number_active,
    #  switchoff_freq, guarantor_reachable_pct,
    #  ai_calls_total, ai_calls_paid,
    #  tele_calls_total, tele_ptp_given, tele_ptp_honored,
    #  field_visits_total, field_paid,
    #  disposition, fraud_flag)
    ("LN-0012","Rajesh Kumar",       2, 94, 0, 0, "UPI_AUTOPAY",  88, 90, 1, 0, 70, 4,4, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0047","Priya Sharma",       1, 97, 0, 0, "ENACH",        95, 95, 1, 0, 80, 5,5, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0083","Amit Singh",         4, 85, 1, 0, "BBPS",         76, 80, 1, 1, 65, 3,2, 1,1,1, 0,0, "cooperative",    False),
    ("LN-0115","Sunita Patel",       3, 90, 0, 0, "UPI_AUTOPAY",  82, 85, 1, 0, 72, 6,5, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0129","Vikram Nair",        5, 78, 1, 0, "BBPS",         70, 75, 1, 1, 60, 4,3, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0144","Kavitha Rao",        2, 92, 0, 0, "ENACH",        90, 88, 1, 0, 75, 5,5, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0156","Deepak Mehta",       6, 75, 2, 1, "BBPS",         65, 70, 1, 1, 55, 3,2, 2,2,1, 1,0, "cooperative",    False),
    ("LN-0168","Ananya Iyer",        3, 88, 1, 0, "UPI_AUTOPAY",  79, 82, 1, 0, 68, 4,3, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0172","Suresh Gupta",       7, 72, 2, 1, "BBPS",         60, 65, 1, 2, 50, 2,1, 3,2,1, 1,0, "cooperative",    False),
    ("LN-0189","Meena Joshi",        1, 96, 0, 0, "ENACH",        93, 92, 1, 0, 78, 6,6, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0203","Ramesh Tiwari",     12, 60, 3, 1, "QR",           45, 55, 1, 3, 50, 2,0, 3,2,1, 2,1, "excuse_maker",   False),
    ("LN-0218","Lalita Yadav",      10, 65, 2, 1, "BBPS",         50, 60, 1, 2, 45, 3,1, 4,3,1, 1,0, "excuse_maker",   False),
    ("LN-0235","Prakash Verma",     15, 55, 3, 2, "QR",           40, 45, 1, 3, 40, 1,0, 4,3,1, 2,1, "excuse_maker",   False),
    ("LN-0247","Geeta Mishra",       9, 68, 2, 1, "BBPS",         55, 58, 1, 2, 48, 2,1, 3,2,1, 1,0, "excuse_maker",   False),
    ("LN-0261","Naresh Pandey",     11, 62, 3, 1, "QR",           42, 50, 1, 3, 42, 1,0, 3,2,0, 2,1, "excuse_maker",   False),
    ("LN-0279","Sarita Chandra",     8, 70, 2, 1, "BBPS",         52, 62, 1, 2, 46, 3,1, 2,2,1, 1,0, "excuse_maker",   False),
    ("LN-0292","Anil Kumar",        13, 58, 3, 2, "QR",           38, 42, 1, 4, 38, 1,0, 3,2,0, 2,1, "excuse_maker",   False),
    ("LN-0307","Pooja Singh",        7, 73, 2, 1, "BBPS",         58, 65, 1, 2, 52, 3,1, 2,2,1, 0,0, "cooperative",    False),
    ("LN-0319","Harish Agarwal",    14, 56, 4, 2, "QR",           35, 40, 1, 4, 35, 0,0, 3,2,0, 3,2, "pressure_resp",  False),
    ("LN-0334","Rekha Pillai",       9, 66, 2, 1, "BBPS",         48, 55, 1, 3, 44, 2,0, 4,3,1, 1,0, "excuse_maker",   False),
    ("LN-0348","Ganesh Patil",      22, 40, 5, 3, "CASH",         20, 25, 1, 5, 30, 0,0, 1,1,0, 4,3, "pressure_resp",  False),
    ("LN-0362","Shanti Devi",       25, 35, 5, 3, "CASH",         15, 20, 0, 6, 25, 0,0, 0,0,0, 3,2, "avoider",        False),
    ("LN-0377","Mohan Lal",         18, 45, 4, 2, "CASH",         22, 28, 1, 5, 20, 0,0, 1,1,0, 5,4, "pressure_resp",  False),
    ("LN-0391","Champa Bai",        30, 30, 6, 3, "CASH",         10, 15, 0, 7, 15, 0,0, 0,0,0, 4,3, "avoider",        False),
    ("LN-0405","Ratan Singh",       20, 42, 4, 2, "CASH",         18, 22, 1, 6, 18, 0,0, 1,1,0, 3,2, "pressure_resp",  False),
    ("LN-0418","Savita Kumari",     28, 32, 6, 3, "CASH",         12, 18, 0, 7, 12, 0,0, 0,0,0, 5,4, "avoider",        False),
    ("LN-0429","Dinesh Sharma",     35, 25, 7, 4, "CASH",          8, 10, 0, 8, 10, 0,0, 0,0,0, 6,5, "avoider",        True),
    ("LN-0443","Leela Nair",        19, 44, 4, 2, "QR",           25, 30, 1, 5, 22, 0,0, 2,1,0, 3,2, "pressure_resp",  False),
    ("LN-0457","Brijesh Yadav",     32, 28, 6, 3, "CASH",         10, 12, 0, 8,  8, 0,0, 0,0,0, 5,4, "avoider",        False),
    ("LN-0462","Usha Tiwari",       24, 38, 5, 3, "CASH",         14, 20, 0, 6, 14, 0,0, 0,0,0, 4,3, "avoider",        False),
    ("LN-0478","Mahesh Gupta",      16, 52, 3, 2, "QR",           35, 40, 1, 4, 32, 1,0, 3,2,0, 2,1, "excuse_maker",   False),
    ("LN-0491","Seema Joshi",       21, 42, 4, 2, "CASH",         20, 26, 1, 5, 18, 0,0, 1,1,0, 3,2, "pressure_resp",  False),
    ("LN-0504","Pankaj Mishra",     17, 50, 3, 2, "QR",           38, 44, 1, 3, 34, 1,0, 3,2,0, 2,1, "excuse_maker",   False),
    ("LN-0518","Jyoti Verma",       26, 36, 5, 3, "CASH",         15, 20, 0, 7, 12, 0,0, 0,0,0, 4,3, "avoider",        False),
    ("LN-0531","Rohit Pandey",      29, 30, 6, 3, "CASH",         12, 16, 0, 7, 10, 0,0, 0,0,0, 5,4, "avoider",        False),
    ("LN-0547","Kamala Rao",        33, 26, 7, 4, "CASH",          9, 12, 0, 9,  8, 0,0, 0,0,0, 6,5, "avoider",        True),
    ("LN-0558","Sanjay Patil",       6, 77, 1, 0, "BBPS",         66, 70, 1, 1, 58, 3,1, 2,2,1, 1,0, "cooperative",    False),
    ("LN-0572","Neha Chandra",       4, 83, 1, 0, "UPI_AUTOPAY",  80, 82, 1, 0, 68, 4,3, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0589","Ajay Kumar",         8, 70, 2, 1, "BBPS",         55, 60, 1, 2, 50, 2,1, 2,2,1, 1,0, "cooperative",    False),
    ("LN-0603","Sangeeta Singh",    11, 63, 3, 1, "QR",           42, 48, 1, 3, 40, 1,0, 3,2,1, 2,1, "excuse_maker",   False),
    ("LN-0617","Vivek Agarwal",      5, 80, 1, 0, "BBPS",         72, 76, 1, 1, 62, 4,3, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0628","Archana Pillai",     7, 74, 2, 1, "BBPS",         63, 68, 1, 1, 55, 3,2, 1,1,1, 0,0, "cooperative",    False),
    ("LN-0642","Sunil Patil",       13, 58, 3, 2, "QR",           36, 42, 1, 4, 35, 1,0, 3,2,0, 2,1, "excuse_maker",   False),
    ("LN-0656","Madhuri Devi",      27, 34, 6, 3, "CASH",         13, 18, 0, 7, 10, 0,0, 0,0,0, 4,3, "avoider",        False),
    ("LN-0671","Biswas Roy",        10, 67, 2, 1, "BBPS",         52, 58, 1, 2, 46, 2,1, 2,2,1, 1,0, "excuse_maker",   False),
    ("LN-0684","Tara Singh",        38, 20, 8, 4, "CASH",          6,  8, 0,10,  5, 0,0, 0,0,0, 7,5, "avoider",        True),
    ("LN-0698","Renu Sharma",        3, 91, 0, 0, "UPI_AUTOPAY",  87, 89, 1, 0, 74, 5,5, 0,0,0, 0,0, "cooperative",    False),
    ("LN-0712","Kishore Gupta",      9, 68, 2, 1, "QR",           46, 54, 1, 2, 42, 2,0, 3,2,1, 1,0, "excuse_maker",   False),
]

COLUMNS = [
    "loan_id","name","avg_delay_days","on_time_pct","bounce_freq","missed_emi",
    "payment_mode","call_pickup_rate","ptp_honored_pct","number_active",
    "switchoff_freq","guarantor_reachable_pct",
    "ai_calls_total","ai_calls_paid",
    "tele_calls_total","tele_ptp_given","tele_ptp_honored",
    "field_visits_total","field_paid",
    "disposition","fraud_flag"
]

df = pd.DataFrame(RAW_DATA, columns=COLUMNS)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Payment mode digital affinity score ─────────────────────────────────
MODE_SCORE = {
    "UPI_AUTOPAY": 100,
    "ENACH":        90,
    "BBPS":         75,
    "QR":           45,
    "CASH":         15,
}
df["mode_score"] = df["payment_mode"].map(MODE_SCORE)


# ── 2b. Contactability Score (0–100) ─────────────────────────────────────────
#  Weights: pickup 30%, ptp_honored 25%, number_active 15%,
#           guarantor 10%, switchoff_penalty 20%
def compute_contactability(row) -> int:
    pickup_w     = row["call_pickup_rate"]   * 0.30
    ptp_w        = row["ptp_honored_pct"]    * 0.25
    active_w     = row["number_active"]      * 15          # binary → 0 or 15
    guar_w       = row["guarantor_reachable_pct"] * 0.10
    switchoff_p  = min(row["switchoff_freq"] * 5, 30)      # cap penalty at 30
    raw = pickup_w + ptp_w + active_w + guar_w - switchoff_p
    return max(0, min(100, round(raw)))

df["contactability_score"] = df.apply(compute_contactability, axis=1)


# ── 2c. Delay severity score (inverse — higher score = less delay) ───────────
df["delay_score"] = (100 - df["avg_delay_days"] * 2.5).clip(lower=0).round().astype(int)


# ── 2d. Channel Affinity ─────────────────────────────────────────────────────
df["ai_affinity"]    = ((df["ai_calls_paid"]   / df["ai_calls_total"].replace(0, 1)) * 100).round().astype(int)
df["field_affinity"] = ((df["field_paid"]       / df["field_visits_total"].replace(0, 1)) * 100).round().astype(int)
df["tele_ptp_rate"]  = ((df["tele_ptp_honored"] / df["tele_ptp_given"].replace(0, 1)) * 100).round().astype(int)


# ── 2e. Field Visit Necessity Score (0–100) ──────────────────────────────────
def field_necessity(row) -> int:
    score = 0
    if row["call_pickup_rate"] < 30:        score += 25
    if row["number_active"] == 0:            score += 20
    if row["payment_mode"] == "CASH":        score += 20
    if row["avg_delay_days"] > 20:           score += 15
    if row["guarantor_reachable_pct"] < 25:  score += 10
    if row["switchoff_freq"] >= 5:           score += 10
    return min(100, score)

df["field_necessity_score"] = df.apply(field_necessity, axis=1)


# ── 2f. Escalation Resistance Score ─────────────────────────────────────────
def escalation_resistance(row) -> int:
    score = 0
    if row["field_visits_total"] > 2:               score += 30
    if row["tele_ptp_honored"] < row["tele_ptp_given"] / max(row["tele_ptp_given"],1) * 50:
                                                     score += 25
    if row["avg_delay_days"] > 15:                   score += 25
    if row["ai_affinity"] < 30 and row["ai_calls_total"] > 1:
                                                     score += 20
    return min(100, score)

df["escalation_resistance"] = df.apply(escalation_resistance, axis=1)


# ── 2g. Composite Overall Score ──────────────────────────────────────────────
df["composite_score"] = (
    df["contactability_score"] * 0.40 +
    df["delay_score"]          * 0.25 +
    df["on_time_pct"]          * 0.20 +
    (100 - df["bounce_freq"] * 15).clip(lower=0) * 0.10 +
    df["mode_score"]           * 0.05
).round().astype(int)


# ── 2h. Customer Persona ─────────────────────────────────────────────────────
def assign_persona(row) -> str:
    cs = row["contactability_score"]
    if row["on_time_pct"] >= 85 and row["avg_delay_days"] <= 5:
        return "Digital Disciplined"
    if cs < 25 and row["avg_delay_days"] > 20:
        return "Chronic Avoider"
    if row["payment_mode"] == "CASH" and row["avg_delay_days"] > 15:
        return "Cash Dependent"
    if row["ptp_honored_pct"] < 40 and row["avg_delay_days"] > 10:
        return "Broken PTP"
    if row["avg_delay_days"] > 10 and cs > 40:
        return "Soft Delinquent"
    if row["guarantor_reachable_pct"] < 25 and cs < 35:
        return "Guarantor-Driven"
    return "Excuse Maker"

df["persona"] = df.apply(assign_persona, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ALLOCATION ENGINE  (Level 1 — Rule-Based)
# ─────────────────────────────────────────────────────────────────────────────

def allocate_channel(row) -> str:
    """
    Core decision logic — returns primary channel.

    Priority rules (evaluated top-down, first match wins):
      1. Hard field triggers  → Field Visit
      2. Historical affinity  → honour what worked before
      3. Score-based routing  → AI / Tele / Field
    """
    cs      = row["contactability_score"]
    delay   = row["avg_delay_days"]
    mode    = row["payment_mode"]
    active  = row["number_active"]
    ai_aff  = row["ai_affinity"]
    fld_aff = row["field_affinity"]
    fraud   = row["fraud_flag"]

    # ── Hard triggers → Field Visit ─────────────────────────────────────────
    if fraud:                                          return "Field Visit"
    if active == 0:                                    return "Field Visit"
    if mode == "CASH" and delay > 10:                  return "Field Visit"
    if cs < 25:                                        return "Field Visit"
    if delay > 30:                                     return "Field Visit"

    # ── Historical channel affinity ──────────────────────────────────────────
    if fld_aff >= 60 and row["field_visits_total"] >= 2:
        return "Field Visit"    # customer responds only to field
    if ai_aff >= 60 and row["ai_calls_total"] >= 2:
        return "AI Calling"     # customer has responded well to AI

    # ── Score-based routing ──────────────────────────────────────────────────
    if cs >= 65 and delay <= 7 and row["mode_score"] >= 75:
        return "AI Calling"

    if cs >= 45 and delay <= 15:
        return "Telecalling"

    if delay > 15 or mode in ("QR", "CASH") or cs < 45:
        return "Field Visit"

    return "Telecalling"   # fallback


def assign_priority(row) -> str:
    if row["avg_delay_days"] > 25 or row["contactability_score"] < 20:
        return "High"
    if row["avg_delay_days"] > 10 or row["contactability_score"] < 45:
        return "Medium"
    return "Low"


def escalation_path(row) -> str:
    """Which stage would this case escalate to if unresolved?"""
    if row["allocated_channel"] == "AI Calling":
        return "AI → Telecalling → Field Visit"
    if row["allocated_channel"] == "Telecalling":
        return "Telecalling → Field Visit"
    return "Field Visit (terminal)"


df["allocated_channel"] = df.apply(allocate_channel, axis=1)
df["priority"]          = df.apply(assign_priority, axis=1)
df["escalation_path"]   = df.apply(escalation_path, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROBABILITY ESTIMATION  (heuristic, not ML)
# ─────────────────────────────────────────────────────────────────────────────

def channel_probabilities(row):
    """
    Estimate success probability for each channel.
    Returns dict {channel: probability_pct}
    """
    cs = row["contactability_score"]
    ms = row["mode_score"]
    delay = row["avg_delay_days"]

    ai_prob    = int(min(95, cs * 0.55 + ms * 0.30 + max(0, 30 - delay) * 0.6))
    tele_prob  = int(min(90, cs * 0.40 + 25 + max(0, 20 - delay) * 0.5))
    field_prob = int(min(88, 55 + (100 - cs) * 0.25 + min(delay * 0.5, 20)))

    return {"AI Calling": ai_prob, "Telecalling": tele_prob, "Field Visit": field_prob}

df["channel_probs"] = df.apply(channel_probabilities, axis=1)
df["ai_prob"]       = df["channel_probs"].apply(lambda x: x["AI Calling"])
df["tele_prob"]     = df["channel_probs"].apply(lambda x: x["Telecalling"])
df["field_prob"]    = df["channel_probs"].apply(lambda x: x["Field Visit"])
df.drop(columns=["channel_probs"], inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. REPORTING
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_COLS = [
    "loan_id", "name", "avg_delay_days", "payment_mode",
    "contactability_score", "composite_score",
    "persona", "allocated_channel", "priority",
    "ai_prob", "tele_prob", "field_prob",
    "escalation_path"
]

def print_section(title: str) -> None:
    width = 80
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def run_reports(df: pd.DataFrame) -> None:

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("ALLOCATION SUMMARY")
    counts = df["allocated_channel"].value_counts()
    total  = len(df)
    for ch, cnt in counts.items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 3)
        print(f"  {ch:<18} {cnt:>3} cases  ({pct:5.1f}%)  {bar}")
    print(f"\n  Total cases: {total}")

    # ── Priority breakdown ────────────────────────────────────────────────────
    print_section("PRIORITY BREAKDOWN")
    pivot = df.groupby(["allocated_channel","priority"]).size().unstack(fill_value=0)
    print(tabulate(pivot, headers="keys", tablefmt="rounded_outline"))

    # ── Persona distribution ──────────────────────────────────────────────────
    print_section("PERSONA DISTRIBUTION")
    persona_counts = df["persona"].value_counts()
    for p, cnt in persona_counts.items():
        print(f"  {p:<25} {cnt:>3} borrowers")

    # ── Full allocation roster ────────────────────────────────────────────────
    print_section("FULL ALLOCATION ROSTER")
    display_df = df[DISPLAY_COLS].copy()
    display_df.columns = [
        "Loan ID","Name","Delay(d)","Mode",
        "Contact","Composite",
        "Persona","Channel","Priority",
        "AI%","Tele%","Field%",
        "Escalation Path"
    ]
    print(tabulate(display_df, headers="keys", tablefmt="rounded_outline",
                   showindex=False, maxcolwidths=22))

    # ── High priority cases ───────────────────────────────────────────────────
    print_section("HIGH PRIORITY CASES — IMMEDIATE ACTION REQUIRED")
    high = df[df["priority"] == "High"][["loan_id","name","avg_delay_days",
                                          "contactability_score","allocated_channel",
                                          "escalation_path"]]
    print(tabulate(high, headers=["Loan ID","Name","Delay","Contactability",
                                   "Channel","Escalation Path"],
                   tablefmt="rounded_outline", showindex=False))

    # ── Field visit necessity ─────────────────────────────────────────────────
    print_section("TOP 10 — FIELD VISIT NECESSITY SCORE")
    top_field = df.nlargest(10, "field_necessity_score")[
        ["loan_id","name","payment_mode","contactability_score",
         "field_necessity_score","allocated_channel"]
    ]
    print(tabulate(top_field, headers=["Loan ID","Name","Mode","Contactability",
                                        "Field Necessity","Channel"],
                   tablefmt="rounded_outline", showindex=False))

    # ── Avg scores per channel ────────────────────────────────────────────────
    print_section("AVERAGE SCORES BY ALLOCATED CHANNEL")
    avg_scores = df.groupby("allocated_channel").agg(
        Avg_Contactability  = ("contactability_score","mean"),
        Avg_Composite       = ("composite_score","mean"),
        Avg_Delay_Days      = ("avg_delay_days","mean"),
        Avg_FieldNecessity  = ("field_necessity_score","mean"),
        Count               = ("loan_id","count")
    ).round(1)
    print(tabulate(avg_scores, headers="keys", tablefmt="rounded_outline"))

    # ── Escalation resistance ─────────────────────────────────────────────────
    print_section("TOP 10 — HIGHEST ESCALATION RESISTANCE (CHRONIC CASES)")
    chronic = df.nlargest(10, "escalation_resistance")[
        ["loan_id","name","avg_delay_days","contactability_score",
         "escalation_resistance","persona","allocated_channel"]
    ]
    print(tabulate(chronic, headers=["Loan ID","Name","Delay","Contactability",
                                      "Resistance","Persona","Channel"],
                   tablefmt="rounded_outline", showindex=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. SINGLE CUSTOMER LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def lookup(loan_id: str) -> None:
    """Detailed view of a single borrower's allocation decision."""
    row = df[df["loan_id"] == loan_id]
    if row.empty:
        print(f"  Loan ID {loan_id} not found.")
        return
    r = row.iloc[0]
    print_section(f"CASE DETAIL — {r['loan_id']} / {r['name']}")
    fields = [
        ("Payment mode",         r["payment_mode"]),
        ("Avg delay (days)",     r["avg_delay_days"]),
        ("On-time payment %",    r["on_time_pct"]),
        ("Bounce frequency",     r["bounce_freq"]),
        ("Missed EMIs",          r["missed_emi"]),
        ("Call pick-up rate",    f"{r['call_pickup_rate']}%"),
        ("PTP honored %",        f"{r['ptp_honored_pct']}%"),
        ("Number active",        "Yes" if r["number_active"] else "No"),
        ("Switchoff frequency",  r["switchoff_freq"]),
        ("Guarantor reachable",  f"{r['guarantor_reachable_pct']}%"),
        ("─── SCORES ───────────",""),
        ("Contactability score", r["contactability_score"]),
        ("Mode score",           r["mode_score"]),
        ("Delay score",          r["delay_score"]),
        ("Composite score",      r["composite_score"]),
        ("Field necessity score",r["field_necessity_score"]),
        ("Escalation resistance",r["escalation_resistance"]),
        ("AI affinity",          f"{r['ai_affinity']}%"),
        ("Field affinity",       f"{r['field_affinity']}%"),
        ("─── DECISION ─────────",""),
        ("Persona",              r["persona"]),
        ("Allocated channel",    r["allocated_channel"]),
        ("Priority",             r["priority"]),
        ("AI success prob",      f"{r['ai_prob']}%"),
        ("Tele success prob",    f"{r['tele_prob']}%"),
        ("Field success prob",   f"{r['field_prob']}%"),
        ("Escalation path",      r["escalation_path"]),
        ("Fraud flag",           "YES ⚠" if r["fraud_flag"] else "No"),
    ]
    for k, v in fields:
        if str(k).startswith("───"):
            print(f"\n  {k}")
        else:
            print(f"  {k:<28} {v}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ESCALATION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_escalation(loan_id: str) -> None:
    """
    Print the multi-stage escalation plan for a given loan.
    Each stage shows triggers, max attempts, and pass-fail conditions.
    """
    row = df[df["loan_id"] == loan_id]
    if row.empty:
        return
    r = row.iloc[0]
    ch = r["allocated_channel"]
    print_section(f"ESCALATION PLAN — {loan_id} ({r['name']})")

    stages = []
    if ch == "AI Calling":
        stages = [
            ("Stage 1 — AI Calling",
             "Contactability ≥ 65, digital mode, delay ≤ 7d",
             "3 attempts over 48 hrs via IVR + WhatsApp",
             "Payment received OR PTP given and honored"),
            ("Stage 2 — Telecalling",
             "AI unresolved after 3 attempts",
             "5 calls over 5 days, human negotiation",
             "PTP honored OR payment received"),
            ("Stage 3 — Field Visit",
             "Telecalling unresolved",
             "2 visits, physical collection",
             "Payment collected on site"),
        ]
    elif ch == "Telecalling":
        stages = [
            ("Stage 1 — Telecalling",
             "Medium contactability, moderate delay",
             "5 calls over 5 days",
             "PTP honored OR payment received"),
            ("Stage 2 — Field Visit",
             "Tele unresolved OR broken PTP",
             "2 visits, physical collection",
             "Payment collected on site"),
        ]
    else:
        stages = [
            ("Stage 1 — Field Visit",
             "Low contactability / cash / inactive number / fraud",
             "Up to 3 visits; escalate to legal after",
             "Payment collected on site"),
        ]

    for title, trigger, action, success in stages:
        print(f"\n  ┌─ {title}")
        print(f"  │  Trigger : {trigger}")
        print(f"  │  Action  : {action}")
        print(f"  └─ Success : {success}")
        if stages.index((title,trigger,action,success)) < len(stages)-1:
            print("       ↓ if unresolved")


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(path: str = "allocation_output.csv") -> None:
    export_cols = [
        "loan_id","name","avg_delay_days","payment_mode",
        "contactability_score","composite_score","field_necessity_score",
        "escalation_resistance","ai_affinity","field_affinity",
        "persona","allocated_channel","priority",
        "ai_prob","tele_prob","field_prob","escalation_path","fraud_flag"
    ]
    df[export_cols].to_csv(path, index=False)
    print(f"\n  ✓ Allocation output exported to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "▓" * 80)
    print("  NBFC COLLECTION ALLOCATION ENGINE  — v1.0")
    print("  Collection Channel Routing  |  48 Borrowers  |  May 2026")
    print("▓" * 80)

    # Full reports
    run_reports(df)

    # Sample single-borrower lookups
    lookup("LN-0012")   # Digital disciplined — should get AI Calling
    lookup("LN-0391")   # Chronic avoider — should get Field Visit
    lookup("LN-0203")   # Excuse maker — should get Telecalling

    # Sample escalation plans
    simulate_escalation("LN-0083")
    simulate_escalation("LN-0348")

    # Export
    export_csv("allocation_output.csv")

    print("\n" + "▓" * 80)
    print("  ENGINE RUN COMPLETE")
    print("▓" * 80 + "\n")
