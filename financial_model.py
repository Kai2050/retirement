"""
================================================================================
HOUSEHOLD FINANCIAL LIFE MODEL
================================================================================
Models cashflow year-by-year from current age through retirement until death.
Compares two scenarios:
  Scenario A: Continue paying mortgage until it expires
  Scenario B: Pay off mortgage immediately with available cash/investments

Edit all inputs in the INPUTS section below, then run the script.
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ================================================================================
# INPUTS — Edit everything in this section before running
# ================================================================================

# --- Personal / Timeline --------------------------------------------------------
age_now           = 53      # Current age (primary earner / reference age)
age_retirement    = 60      # Age at which both partners stop working
age_death         = 85      # Age at which the model ends
calendar_year_now = 2026    # Current calendar year (used to key one-off costs)

# --- Investments & Cash ---------------------------------------------------------
current_investments  = 2_429_000  # Total investment portfolio value today ($)
investment_gain_rate =       6.0  # Annual investment return (%)
inflation_rate       =       2.5  # Annual inflation rate (%)

# --- Tax ------------------------------------------------------------------------
tax_rate_working    = 20.0   # Effective income tax rate while working (%)
tax_rate_retirement = 20.0   # Effective income tax rate in retirement (%)
tax_rate_investment =  15.0   # Capital gains / investment tax rate (%)
salt_deduction      = 14_000 # SALT deduction subtracted from gross income before tax ($)

# --- Employment Income (working years only) -------------------------------------
income_nik = 220_000   # Nik's annual gross income ($)
income_og  = 66_000   # OG's annual gross income ($)
# total_income is calculated automatically

# --- Property / Housing Costs ---------------------------------------------------
current_house_value      = 2_400_000  # Current estimated home value ($)
house_price_growth_rate  =       2.0  # Annual house price appreciation (%)
remaining_mortgage       =   806_000  # Remaining mortgage balance ($)
monthly_mortgage_payment =     6_532  # Monthly repayment ($)
mortgage_years_remaining =        15  # Years left on the mortgage
yearly_property_tax      =    30_000  # Annual property tax ($)
yearly_home_insurance    =    11_000  # Annual home insurance ($)

# --- Living Expenses ------------------------------------------------------------
basic_spending       = 78_000  # Annual spending excl. house costs & child costs — inflates every year ($)
child_costs          = 60_000  # Annual child costs — WORKING YEARS ONLY ($)
medical_costs_retirement = 20_000  # Annual medical costs — RETIREMENT ONLY ($)

# --- US Income Streams (activate at age 62) ------------------------------------
ss_nik_monthly         = 1_858  # Nik's US Social Security per month ($)
ss_og_monthly          =   982  # OG's US Social Security per month ($)
pension_nik_us_monthly =   857  # Nik's US pension per month ($)

# --- UK Income Streams (activate at age 67) ------------------------------------
pension_nik_uk_monthly = 308  # Nik's UK pension per month ($)
pension_og_uk_monthly  =   0  # OG's UK pension per month ($)

# --- Pension / Other Debt -------------------------------------------------------
mpeg_balance = 100_000   # Loan to be repaid — deducted from portfolio in year 1 ($)

# --- One-Off Costs --------------------------------------------------------------
# Keyed by CALENDAR YEAR. NOT inflation-adjusted. Use {} if none.
one_off_costs = {
    2026: 50_000,   # car
    2028: 40_000,   # college costs
    2029: 40_000,   # college costs
    2030: 40_000,   # college costs
    2031: 40_000,   # college costs
}

# ================================================================================
# DERIVED VALUES — Do not edit below this line
# ================================================================================

total_income = income_nik + income_og
investable_start = current_investments


# ================================================================================
# MORTGAGE HELPER — back-solve for implied annual interest rate
# ================================================================================

def solve_mortgage_rate(balance, monthly_pmt, years):
    """Newton-Raphson solver for the implied monthly interest rate."""
    n = years * 12
    if monthly_pmt <= 0 or balance <= 0 or n <= 0:
        return 0.0
    r = 0.005  # initial monthly rate guess ~6% annual
    for _ in range(2000):
        factor = (1 + r) ** n
        denom  = factor - 1
        if abs(denom) < 1e-14:
            break
        f  = balance * r * factor / denom - monthly_pmt
        df = balance * (factor * (n * r + 1) - n * r - factor) / (denom ** 2)
        if abs(df) < 1e-14:
            break
        r_new = r - f / df
        r_new = max(r_new, 1e-8)          # keep positive
        if abs(r_new - r) < 1e-10:
            r = r_new
            break
        r = r_new
    return r  # monthly rate


monthly_mortgage_rate = solve_mortgage_rate(
    remaining_mortgage, monthly_mortgage_payment, mortgage_years_remaining
)
annual_mortgage_rate  = monthly_mortgage_rate * 12


def amortise_year(balance, monthly_rate, monthly_pmt, months=12):
    """
    Simulate up to `months` mortgage payments.
    Returns (new_balance, annual_payment_made, interest_portion).
    """
    total_paid    = 0.0
    total_interest = 0.0
    for _ in range(months):
        if balance <= 0:
            break
        interest  = balance * monthly_rate
        principal = monthly_pmt - interest
        principal = max(0.0, min(principal, balance))   # clamp
        balance   = max(0.0, balance - principal)
        total_paid     += interest + principal
        total_interest += interest
    return balance, total_paid, total_interest


# ================================================================================
# CORE MODEL FUNCTION
# ================================================================================

def run_model(scenario_name, pay_off_mortgage):
    """Run year-by-year simulation. Returns (DataFrame, summary_dict)."""

    rows = []

    # Starting state
    portfolio    = float(investable_start)
    cash_balance = 0.0

    # Deduct MPEG debt from portfolio immediately on day 1
    portfolio -= float(mpeg_balance)

    if pay_off_mortgage:
        # Lump-sum payoff from the investment portfolio on day 1
        portfolio -= float(remaining_mortgage)
        mort_bal   = 0.0
        mort_months = 0
    else:
        mort_bal    = float(remaining_mortgage)
        mort_months = mortgage_years_remaining * 12

    inf_factor = 1.0   # cumulative inflation (1.0 in year 0 = no adjustment yet)

    for i in range(age_death - age_now + 1):

        age  = age_now + i
        year = calendar_year_now + i

        if i > 0:
            inf_factor *= (1.0 + inflation_rate / 100.0)

        is_working = (age < age_retirement)
        phase      = "Working" if is_working else "Retired"

        # ---- INFLOWS -----------------------------------------------------------

        gross_income = total_income if is_working else 0.0  # NOT inflation-adjusted

        # Investment gains only if portfolio is positive
        inv_gains = max(0.0, portfolio) * (investment_gain_rate / 100.0)

        ss_income  = (ss_nik_monthly + ss_og_monthly)  * 12 * inf_factor if age >= 62 else 0.0
        pension_us = pension_nik_us_monthly             * 12 * inf_factor if age >= 62 else 0.0
        pension_uk = (pension_nik_uk_monthly + pension_og_uk_monthly) * 12 * inf_factor if age >= 67 else 0.0

        # Investment gains included in total inflows
        total_inflows = gross_income + inv_gains + ss_income + pension_us + pension_uk

        # ---- OUTFLOWS ----------------------------------------------------------

        # Income tax
        if is_working and gross_income > 0:
            taxable    = max(0.0, gross_income - salt_deduction * inf_factor)
            income_tax = taxable * (tax_rate_working / 100.0)
        elif not is_working and (ss_income + pension_us + pension_uk) > 0:
            ret_income = ss_income + pension_us + pension_uk
            income_tax = ret_income * (tax_rate_retirement / 100.0)
        else:
            income_tax = 0.0

        # Tax on investment gains
        investment_tax = inv_gains * (tax_rate_investment / 100.0)

        # Mortgage amortisation
        if mort_months > 0:
            months_now    = min(12, mort_months)
            mort_bal, annual_pmt, _ = amortise_year(
                mort_bal, monthly_mortgage_rate, monthly_mortgage_payment, months_now
            )
            mort_months  = max(0, mort_months - 12)
            if mort_months == 0:
                mort_bal = 0.0
        else:
            annual_pmt = 0.0

        # Property costs — tax is FIXED (no inflation), insurance inflated
        prop_tax  = yearly_property_tax               # fixed — no inflation
        home_ins  = yearly_home_insurance * inf_factor

        # Living expenses — basic_spending inflates every year regardless of phase
        basic_spend = basic_spending * inf_factor
        if is_working:
            child_applied   = child_costs * inf_factor
            medical_applied = 0.0
        else:
            child_applied   = 0.0
            medical_applied = medical_costs_retirement * inf_factor

        # One-off costs (not inflated)
        one_off = float(one_off_costs.get(year, 0))

        total_outflows = (
            income_tax + investment_tax
            + annual_pmt
            + prop_tax + home_ins
            + basic_spend + child_applied + medical_applied
            + one_off
        )

        # ---- BALANCES ----------------------------------------------------------

        # Simple: port_end = port_start + total_inflows - total_outflows
        net_cashflow  = total_inflows - total_outflows

        portfolio_start = portfolio
        portfolio       = portfolio_start + net_cashflow

        # House value appreciates at house_price_growth_rate annually
        house_value   = current_house_value * ((1 + house_price_growth_rate / 100.0) ** i)

        # House equity = house value minus remaining mortgage balance
        house_equity  = house_value - mort_bal

        # Net worth = investment portfolio + house equity (not full house value)
        net_worth     = portfolio + house_equity

        # Real net worth = net worth in today's dollars (deflated by cumulative inflation)
        real_net_worth = net_worth / inf_factor

        # ---- RECORD ROW --------------------------------------------------------
        rows.append({
            "year":                   year,
            "age":                    age,
            "phase":                  phase,
            "investments_start_year": round(portfolio_start),
            "gross_income":           round(gross_income),
            "investment_gains":       round(inv_gains),
            "ss_income":              round(ss_income),
            "pension_income_us":      round(pension_us),
            "pension_income_uk":      round(pension_uk),
            "total_inflows":          round(total_inflows),
            "income_tax":             round(income_tax),
            "investment_tax":         round(investment_tax),
            "mortgage_payment":       round(annual_pmt),
            "property_tax":           round(prop_tax),
            "home_insurance":         round(home_ins),
            "basic_spending":         round(basic_spend),
            "child_costs_applied":    round(child_applied),
            "medical_costs_applied":  round(medical_applied),
            "one_off_costs_applied":  round(one_off),
            "total_outflows":         round(total_outflows),
            "net_cashflow":           round(net_cashflow),
            "investment_portfolio":   round(portfolio),
            "house_value":            round(house_value),
            "mortgage_remaining":     round(mort_bal),
            "house_equity":           round(house_equity),
            "net_worth":              round(net_worth),
            "real_net_worth":          round(real_net_worth),
        })

    df = pd.DataFrame(rows)

    # Summary
    ret_row   = df[df["age"] == age_retirement]
    death_row = df.iloc[-1]
    summary = {
        "scenario":              scenario_name,
        "wealth_at_retirement":  int(ret_row["investment_portfolio"].values[0]) if len(ret_row) else 0,
        "wealth_at_death":       int(death_row["investment_portfolio"]),
        "total_mortgage_paid":   int(df["mortgage_payment"].sum()),
        "final_portfolio":       int(death_row["investment_portfolio"]),
        "total_one_off_costs":   int(df["one_off_costs_applied"].sum()),
        "total_tax_paid":        int((df["income_tax"] + df["investment_tax"]).sum()),
        "total_net_cashflow":    int(df["net_cashflow"].sum()),
    }
    return df, summary


# ================================================================================
# RUN
# ================================================================================

print("=" * 72)
print("  HOUSEHOLD FINANCIAL LIFE MODEL")
print("=" * 72)
print(f"  Implied mortgage annual rate  : {annual_mortgage_rate * 100:.2f}%")
print(f"  Net investable portfolio      : ${investable_start:,.0f}")
print(f"  Scenario B: portfolio after   ")
print(f"    mortgage payoff             : ${investable_start - remaining_mortgage:,.0f}")
print(f"  Inflation rate                : {inflation_rate}%")
print(f"  Investment return             : {investment_gain_rate}%")
print(f"  Model period                  : Age {age_now} → {age_death}  ({age_death - age_now + 1} yrs)")
print()

df_a, sum_a = run_model("A — Keep Mortgage",    pay_off_mortgage=False)
df_b, sum_b = run_model("B — Pay Off Mortgage", pay_off_mortgage=True)

# ================================================================================
# SUMMARY TABLE
# ================================================================================

def fmt_money(v):
    if isinstance(v, (int, float)):
        sign = "-" if v < 0 else " "
        return f"{sign}${abs(v):>13,.0f}"
    return f" {str(v):>14}"

metrics = [
    ("wealth_at_retirement", "Wealth at retirement"),
    ("wealth_at_death",      "Wealth at death"),
    ("total_mortgage_paid",  "Total mortgage payments"),
    ("final_portfolio",      "Final investment portfolio"),
    ("total_one_off_costs",  "Total one-off costs paid"),
    ("total_tax_paid",       "Total tax paid"),
    ("total_net_cashflow",   "Total net cashflow"),
]

print("=" * 76)
print("  SUMMARY COMPARISON")
print("=" * 76)
print(f"  {'Metric':<35} {'Scenario A':>16} {'Scenario B':>16}  {'Diff (B-A)':>10}")
print("  " + "-" * 72)
for key, label in metrics:
    va, vb = sum_a[key], sum_b[key]
    diff   = vb - va if isinstance(va, (int, float)) else ""
    ds     = ("+" if diff >= 0 else "") + f"${diff:,.0f}" if isinstance(diff, (int, float)) else ""
    print(f"  {label:<35} {fmt_money(va)} {fmt_money(vb)}  {ds:>10}")
print("=" * 76)

# ================================================================================
# DETAILED DATAFRAMES
# ================================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.float_format", "${:,.0f}".format)



# ================================================================================
# EXCEL EXPORT
# ================================================================================

# Two separate Excel files — one per scenario
excel_a = "scenario_A_keep_mortgage.xlsx"
excel_b = "scenario_B_payoff_mortgage.xlsx"
summary_df = pd.DataFrame([sum_a, sum_b])

with pd.ExcelWriter(excel_a, engine="openpyxl") as writer:
    df_a.to_excel(writer, sheet_name="Year_by_Year", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

with pd.ExcelWriter(excel_b, engine="openpyxl") as writer:
    df_b.to_excel(writer, sheet_name="Year_by_Year", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"\n  Excel saved → {excel_a}")
print(f"  Excel saved → {excel_b}")

# ================================================================================
# CHART
# ================================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.patch.set_facecolor("#0d1117")

ref_lines = [
    (age_retirement, f"Retire (age {age_retirement})", "#ffd54f"),
    (62,             "SS + US Pension (62)",            "#ff8a65"),
    (67,             "UK Pension (67)",                  "#ce93d8"),
]

# --- Top: Cumulative Wealth ---
ax1 = axes[0]
ax1.set_facecolor("#161b22")

ax1.plot(df_a["age"], df_a["investment_portfolio"],
         color="#4fc3f7", linewidth=2.5, label="A — Keep Mortgage")
ax1.plot(df_b["age"], df_b["investment_portfolio"],
         color="#81c784", linewidth=2.5, linestyle="--", label="B — Pay Off Mortgage")

ax1.fill_between(df_a["age"],
                 df_a["investment_portfolio"], df_b["investment_portfolio"],
                 where=(df_a["investment_portfolio"] >= df_b["investment_portfolio"]),
                 alpha=0.10, color="#4fc3f7")
ax1.fill_between(df_a["age"],
                 df_a["investment_portfolio"], df_b["investment_portfolio"],
                 where=(df_b["investment_portfolio"] >  df_a["investment_portfolio"]),
                 alpha=0.10, color="#81c784")

ax1.axhline(0, color="#555", linewidth=0.8)
ylo, yhi = ax1.get_ylim()
for ref_age, lbl, col in ref_lines:
    if age_now <= ref_age <= age_death:
        ax1.axvline(ref_age, color=col, linestyle=":", linewidth=1.3, alpha=0.8)
        ax1.text(ref_age + 0.2, ylo + (yhi - ylo) * 0.02,
                 lbl, color=col, fontsize=8, rotation=90, va="bottom")

ax1.set_title("Cumulative Wealth Over Time", color="white", fontsize=13, pad=10)
ax1.set_ylabel("Wealth ($)", color="#aaa", fontsize=10)
ax1.set_xlabel("Age", color="#aaa", fontsize=10)
ax1.tick_params(colors="#aaa")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d", fontsize=10)
ax1.grid(True, color="#21262d", linewidth=0.6)
for sp in ax1.spines.values():
    sp.set_edgecolor("#30363d")

# --- Bottom: Annual Net Cashflow ---
ax2 = axes[1]
ax2.set_facecolor("#161b22")

ages  = df_a["age"].values
cf_a  = df_a["net_cashflow"].values / 1_000
cf_b  = df_b["net_cashflow"].values / 1_000
w     = 0.38

# Positive and negative bars coloured differently
ax2.bar(ages - w/2, [v if v >= 0 else 0 for v in cf_a], width=w,
        color="#4fc3f7", alpha=0.85, label="A — surplus")
ax2.bar(ages - w/2, [v if v <  0 else 0 for v in cf_a], width=w,
        color="#ef5350", alpha=0.85, label="A — deficit")
ax2.bar(ages + w/2, [v if v >= 0 else 0 for v in cf_b], width=w,
        color="#81c784", alpha=0.85, label="B — surplus")
ax2.bar(ages + w/2, [v if v <  0 else 0 for v in cf_b], width=w,
        color="#ff7043", alpha=0.85, label="B — deficit")

ax2.axhline(0, color="#aaa", linewidth=0.8)
ax2.set_title("Annual Net Cashflow", color="white", fontsize=13, pad=10)
ax2.set_ylabel("Net Cashflow ($000s)", color="#aaa", fontsize=10)
ax2.set_xlabel("Age", color="#aaa", fontsize=10)
ax2.tick_params(colors="#aaa")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
ax2.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d",
           fontsize=9, ncol=2)
ax2.grid(True, color="#21262d", linewidth=0.6, axis="y")
for sp in ax2.spines.values():
    sp.set_edgecolor("#30363d")

plt.suptitle(
    f"Financial Life Model  ·  Age {age_now}–{age_death}  ·  Retire at {age_retirement}",
    color="white", fontsize=14, y=1.01, fontweight="bold"
)
plt.tight_layout()

chart_path = "financial_model_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Chart saved  → {chart_path}\n")
plt.show()





# ================================================================================
# HTML VIEWER GENERATION — runs automatically after Excel export
# ================================================================================

import json as _json, base64 as _b64

_drop = ['opening_balance', 'closing_balance', 'cumulative_wealth',
         'mpeg_deducted', 'mortgage_balance']
_df_a_h = df_a.drop(columns=[c for c in _drop if c in df_a.columns])
_df_b_h = df_b.drop(columns=[c for c in _drop if c in df_b.columns])

_data_a = _df_a_h.to_json(orient='records')
_data_b = _df_b_h.to_json(orient='records')

_config = _json.dumps({
    "A": {"inv_gain_rate": investment_gain_rate, "inflation_rate": inflation_rate},
    "B": {"inv_gain_rate": investment_gain_rate, "inflation_rate": inflation_rate}
})

_TEMPLATE_B64 = (
    "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVU"
    "Ri04Ij4KPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwg"
    "aW5pdGlhbC1zY2FsZT0xLjAiPgo8dGl0bGU+SG91c2Vob2xkIEZpbmFuY2lhbCBNb2RlbCDCtyBB"
    "Z2UgNTPigJM4NTwvdGl0bGU+CjxzdHlsZT4KICAqIHsgYm94LXNpemluZzogYm9yZGVyLWJveDsg"
    "bWFyZ2luOiAwOyBwYWRkaW5nOiAwOyB9CgogIGJvZHkgewogICAgYmFja2dyb3VuZDogIzA4MGMx"
    "MjsKICAgIGNvbG9yOiAjY2JkNWUxOwogICAgZm9udC1mYW1pbHk6ICdJQk0gUGxleCBNb25vJywg"
    "J0NvdXJpZXIgTmV3JywgbW9ub3NwYWNlOwogICAgZm9udC1zaXplOiAxMXB4OwogICAgaGVpZ2h0"
    "OiAxMDB2aDsKICAgIGRpc3BsYXk6IGZsZXg7CiAgICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwog"
    "ICAgb3ZlcmZsb3c6IGhpZGRlbjsKICB9CgogIC8qIOKUgOKUgCBIRUFERVIg4pSA4pSAICovCiAg"
    "I2hlYWRlciB7CiAgICBiYWNrZ3JvdW5kOiAjMGEwZjE4OwogICAgYm9yZGVyLWJvdHRvbTogMXB4"
    "IHNvbGlkICMxZTI5M2I7CiAgICBwYWRkaW5nOiAxMHB4IDE2cHg7CiAgICBkaXNwbGF5OiBmbGV4"
    "OwogICAgYWxpZ24taXRlbXM6IGNlbnRlcjsKICAgIGdhcDogMTJweDsKICAgIGZsZXgtd3JhcDog"
    "d3JhcDsKICAgIGZsZXgtc2hyaW5rOiAwOwogIH0KICAjaGVhZGVyIGgxIHsgY29sb3I6ICNlMmU4"
    "ZjA7IGZvbnQtc2l6ZTogMTNweDsgZm9udC13ZWlnaHQ6IDcwMDsgbGV0dGVyLXNwYWNpbmc6IDFw"
    "eDsgfQogICNoZWFkZXIgcCAgeyBjb2xvcjogIzNkNTQ3MDsgZm9udC1zaXplOiA5cHg7IG1hcmdp"
    "bi10b3A6IDJweDsgfQoKICAuc2NlbmFyaW8tYnRuIHsKICAgIHBhZGRpbmc6IDVweCAxNHB4OyBi"
    "b3JkZXItcmFkaXVzOiA1cHg7IGJvcmRlcjogMXB4IHNvbGlkOwogICAgZm9udC1mYW1pbHk6IGlu"
    "aGVyaXQ7IGZvbnQtc2l6ZTogMTFweDsgZm9udC13ZWlnaHQ6IDcwMDsKICAgIGN1cnNvcjogcG9p"
    "bnRlcjsgdHJhbnNpdGlvbjogYWxsIC4xNXM7CiAgfQogIC5zY2VuYXJpby1idG4uYWN0aXZlICB7"
    "IGJvcmRlci1jb2xvcjojNjBhNWZhOyBiYWNrZ3JvdW5kOiMxNjJjNGE7IGNvbG9yOiM5M2M1ZmQ7"
    "IH0KICAuc2NlbmFyaW8tYnRuLmluYWN0aXZleyBib3JkZXItY29sb3I6IzI1MzU0NTsgYmFja2dy"
    "b3VuZDojMGQxNDIwOyBjb2xvcjojM2Q1NDcwOyB9CgogIC8qIOKUgOKUgCBLUEkgU1RSSVAg4pSA"
    "4pSAICovCiAgI2twaXMgewogICAgZGlzcGxheTogZmxleDsgZmxleC1zaHJpbms6IDA7CiAgICBi"
    "b3JkZXItYm90dG9tOiAxcHggc29saWQgIzFlMjkzYjsKICAgIGJhY2tncm91bmQ6ICMwOTBlMTY7"
    "CiAgfQogIC5rcGkgeyBmbGV4OiAxIDEgMTMwcHg7IHBhZGRpbmc6IDdweCAxNHB4OyBib3JkZXIt"
    "cmlnaHQ6IDFweCBzb2xpZCAjMWUyOTNiOyB9CiAgLmtwaS1sYWJlbCB7IGNvbG9yOiAjZmZmOyBm"
    "b250LXNpemU6IDlweDsgbGV0dGVyLXNwYWNpbmc6IDEuMnB4OyBtYXJnaW4tYm90dG9tOiAycHg7"
    "IH0KICAua3BpLXZhbHVlIHsgZm9udC1zaXplOiAxNHB4OyBmb250LXdlaWdodDogNzAwOyB9Cgog"
    "IC8qIOKUgOKUgCBUQUJMRSBXUkFQUEVSIOKUgOKUgCAqLwogICN0YWJsZS13cmFwIHsgZmxleDog"
    "MTsgb3ZlcmZsb3c6IGF1dG87IH0KCiAgdGFibGUgeyBib3JkZXItY29sbGFwc2U6IGNvbGxhcHNl"
    "OyB3aWR0aDogMTAwJTsgdGFibGUtbGF5b3V0OiBhdXRvOyB9CgogIC8qIGdyb3VwIGhlYWRlciBy"
    "b3cgKi8KICAuZ2ggeyBmb250LXNpemU6IDlweDsgZm9udC13ZWlnaHQ6IDgwMDsgbGV0dGVyLXNw"
    "YWNpbmc6IDJweDsgdGV4dC1hbGlnbjogY2VudGVyOwogICAgICAgIHBhZGRpbmc6IDRweCA4cHg7"
    "IHdoaXRlLXNwYWNlOiBub3dyYXA7IH0KICAvKiBjb2x1bW4gaGVhZGVyIHJvdyAqLwogIC5jaCB7"
    "IGZvbnQtc2l6ZTogMTBweDsgZm9udC13ZWlnaHQ6IDYwMDsgdGV4dC1hbGlnbjogcmlnaHQ7IHBh"
    "ZGRpbmc6IDVweCAxMHB4OwogICAgICAgIHdoaXRlLXNwYWNlOiBub3dyYXA7IH0KICB0aGVhZCB7"
    "IHBvc2l0aW9uOiBzdGlja3k7IHRvcDogMDsgei1pbmRleDogMTA7IH0KCiAgLyogZGF0YSBjZWxs"
    "cyAqLwogIHRkIHsgcGFkZGluZzogM3B4IDEwcHg7IHdoaXRlLXNwYWNlOiBub3dyYXA7IH0KICB0"
    "ZC5sYmwgeyB0ZXh0LWFsaWduOiBjZW50ZXI7IH0KICB0ZC5udW0geyB0ZXh0LWFsaWduOiByaWdo"
    "dDsgfQoKICAvKiBncm91cCBjb2xvdXJzICovCiAgLmctaWQgIHsgYmFja2dyb3VuZDojMTExODIy"
    "OyBib3JkZXItY29sb3I6IzI1MzU0NTsgY29sb3I6IzdlYjVjYzsgfQogIC5nLWludiB7IGJhY2tn"
    "cm91bmQ6IzBkMWUzMDsgYm9yZGVyLWNvbG9yOiMxZTNkNWM7IGNvbG9yOiM2MGE1ZmE7IH0KICAu"
    "Zy1pbiAgeyBiYWNrZ3JvdW5kOiMwYTIyMTg7IGJvcmRlci1jb2xvcjojMWE0YTMyOyBjb2xvcjoj"
    "NGFkZTgwOyB9CiAgLmctb3V0IHsgYmFja2dyb3VuZDojMjcwYTBhOyBib3JkZXItY29sb3I6IzUw"
    "MTgxODsgY29sb3I6I2Y4NzE3MTsgfQogIC5nLWNmICB7IGJhY2tncm91bmQ6IzFhMTQwMDsgYm9y"
    "ZGVyLWNvbG9yOiM0YTM4MDA7IGNvbG9yOiNmYmJmMjQ7IH0KICAuZy1zdW0geyBiYWNrZ3JvdW5k"
    "OiMxODBkMzU7IGJvcmRlci1jb2xvcjojMzgyMDZhOyBjb2xvcjojYzA4NGZjOyB9CgogIC5naC5n"
    "LWlkICB7IGJhY2tncm91bmQ6IzFjMmEzYTsgfQogIC5naC5nLWludiB7IGJhY2tncm91bmQ6IzE2"
    "MmM0YTsgfQogIC5naC5nLWluICB7IGJhY2tncm91bmQ6IzEzMzgyODsgfQogIC5naC5nLW91dCB7"
    "IGJhY2tncm91bmQ6IzNkMTIxMjsgfQogIC5naC5nLWNmICB7IGJhY2tncm91bmQ6IzJhMWYwMDsg"
    "fQogIC5naC5nLXN1bSB7IGJhY2tncm91bmQ6IzI4MTU1MDsgfQoKICAuY2guZy1pZCAgeyBiYWNr"
    "Z3JvdW5kOiMxYzJhM2E7IGJvcmRlci1ib3R0b206MnB4IHNvbGlkICMyNTM1NDU7IH0KICAuY2gu"
    "Zy1pbnYgeyBiYWNrZ3JvdW5kOiMxNjJjNGE7IGJvcmRlci1ib3R0b206MnB4IHNvbGlkICMxZTNk"
    "NWM7IH0KICAuY2guZy1pbiAgeyBiYWNrZ3JvdW5kOiMxMzM4Mjg7IGJvcmRlci1ib3R0b206MnB4"
    "IHNvbGlkICMxYTRhMzI7IH0KICAuY2guZy1vdXQgeyBiYWNrZ3JvdW5kOiMzZDEyMTI7IGJvcmRl"
    "ci1ib3R0b206MnB4IHNvbGlkICM1MDE4MTg7IH0KICAuY2guZy1jZiAgeyBiYWNrZ3JvdW5kOiMy"
    "YTFmMDA7IGJvcmRlci1ib3R0b206MnB4IHNvbGlkICM0YTM4MDA7IH0KICAuY2guZy1zdW0geyBi"
    "YWNrZ3JvdW5kOiMyODE1NTA7IGJvcmRlci1ib3R0b206MnB4IHNvbGlkICMzODIwNmE7IH0KCiAg"
    "dHIuaG92IHRkIHsgYmFja2dyb3VuZDogIzFlMjkzYiAhaW1wb3J0YW50OyB9CiAgdHIucmV0aXJl"
    "LWJvcmRlciB0ZCB7IGJvcmRlci10b3A6IDJweCBzb2xpZCAjODU0ZDBlICFpbXBvcnRhbnQ7IH0K"
    "CiAgLyogc3BlY2lhbCBjZWxsIGNvbG91cnMgKi8KICAuYy1udyAgIHsgY29sb3I6I2ZiYmYyNCAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy1jZi1wIHsgY29sb3I6IzRhZGU4MCAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy1jZi1uIHsgY29sb3I6I2Y4NzE3MSAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy10aW4gIHsgY29sb3I6Izg2ZWZhYyAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy10b3V0IHsgY29sb3I6I2ZjYTVhNSAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy1wb3J0IHsgY29sb3I6IzkzYzVmZCAh"
    "aW1wb3J0YW50OyBmb250LXdlaWdodDo3MDA7IH0KICAuYy13b3JrIHsgY29sb3I6IzM0ZDM5OSAh"
    "aW1wb3J0YW50OyB9CiAgLmMtcmV0ICB7IGNvbG9yOiNmYjkyM2MgIWltcG9ydGFudDsgfQoKICAv"
    "KiDilIDilIAgRk9PVEVSIOKUgOKUgCAqLwogICNmb290ZXIgewogICAgcGFkZGluZzogNXB4IDE0"
    "cHg7IGJvcmRlci10b3A6IDFweCBzb2xpZCAjMWUyOTNiOwogICAgY29sb3I6ICMxZTMwNTA7IGZv"
    "bnQtc2l6ZTogOXB4OyB0ZXh0LWFsaWduOiByaWdodDsKICAgIGJhY2tncm91bmQ6ICMwYTBmMTg7"
    "IGZsZXgtc2hyaW5rOiAwOwogIH0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9Imhl"
    "YWRlciI+CiAgPGRpdj4KICAgIDxoMT5IT1VTRUhPTEQgRklOQU5DSUFMIE1PREVMIMK3IEFHRSA1"
    "M+KAkzg1PC9oMT4KICAgIDxwPjYlIHJldHVybiDCtyAxNSUgaW52IHRheCDCtyAyLjUlIGluZmxh"
    "dGlvbiDCtyBSZXRpcmUgNjAgwrcgJDIuNDNNIHN0YXJ0IMK3ICQ3OEsgYmFzZSBzcGVuZCDCtyBG"
    "b3JtdWxhOiBQb3J0IEVuZCA9IFBvcnQgU3RhcnQgKyBUb3RhbCBJbiDiiJIgVG90YWwgT3V0PC9w"
    "PgogIDwvZGl2PgogIDxkaXYgc3R5bGU9Im1hcmdpbi1sZWZ0OmF1dG87ZGlzcGxheTpmbGV4O2dh"
    "cDo2cHg7Ij4KICAgIDxidXR0b24gY2xhc3M9InNjZW5hcmlvLWJ0biBhY3RpdmUiICAgaWQ9ImJ0"
    "bi1BIiBvbmNsaWNrPSJzZXRTY2VuYXJpbygnQScpIj5BIOKAlCBLZWVwIE1vcnRnYWdlPC9idXR0"
    "b24+CiAgICA8YnV0dG9uIGNsYXNzPSJzY2VuYXJpby1idG4gaW5hY3RpdmUiIGlkPSJidG4tQiIg"
    "b25jbGljaz0ic2V0U2NlbmFyaW8oJ0InKSI+QiDigJQgUGF5IE9mZiBNb3J0Z2FnZTwvYnV0dG9u"
    "PgogIDwvZGl2Pgo8L2Rpdj4KCjxkaXYgaWQ9ImtwaXMiPjwvZGl2Pgo8ZGl2IGlkPSJ0YWJsZS13"
    "cmFwIj48dGFibGUgaWQ9InRibCI+PC90YWJsZT48L2Rpdj4KPGRpdiBpZD0iZm9vdGVyIj5BbWJl"
    "ciBib3JkZXIgPSByZXRpcmVtZW50IChhZ2UgNjApIMK3IEhvdmVyIHRvIGhpZ2hsaWdodCByb3c8"
    "L2Rpdj4KCjxzY3JpcHQ+Ci8vIOKUgOKUgCBEQVRBIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgApjb25zdCBEQVRBID0gewogIEE6IF9fREFUQV9BX18sCiAgQjogX19EQVRB"
    "X0JfXwp9Owpjb25zdCBDT05GSUcgPSBfX0NPTkZJR19fOwoKLy8g4pSA4pSAIENPTFVNTiBERUZJ"
    "TklUSU9OUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIAKY29uc3QgQ09MUyA9IFsKICB7azoneWVhcicsICAgICAgICAgICAgICAgICAgIGw6J1ll"
    "YXInLCAgICAgICAgZzonaWQnfSwKICB7azonYWdlJywgICAgICAgICAgICAgICAgICAgIGw6J0Fn"
    "ZScsICAgICAgICAgZzonaWQnfSwKICB7azoncGhhc2UnLCAgICAgICAgICAgICAgICAgIGw6J1Bo"
    "YXNlJywgICAgICAgZzonaWQnfSwKICB7azonaW52ZXN0bWVudHNfc3RhcnRfeWVhcicsIGw6J1Bv"
    "cnQuIFN0YXJ0JywgZzonaW52J30sCiAge2s6J2ludmVzdG1lbnRfZ2FpbnMnLCAgICAgICBsOidJ"
    "bnYgR2FpbnMnLCAgIGc6J2ludid9LAogIHtrOidpbnZlc3RtZW50X3RheCcsICAgICAgICAgbDon"
    "SW52IFRheCcsICAgICBnOidpbnYnfSwKICB7azonaW52ZXN0bWVudF9wb3J0Zm9saW8nLCAgIGw6"
    "J1BvcnQuIEVuZCcsICAgZzonaW52J30sCiAge2s6J2dyb3NzX2luY29tZScsICAgICAgICAgICBs"
    "OidFbXBsb3ltZW50JywgIGc6J2luJ30sCiAge2s6J3NzX2luY29tZScsICAgICAgICAgICAgICBs"
    "OidTb2MgU2VjJywgICAgIGc6J2luJ30sCiAge2s6J3BlbnNpb25faW5jb21lX3VzJywgICAgICBs"
    "OidVUyBQZW5zaW9uJywgIGc6J2luJ30sCiAge2s6J3BlbnNpb25faW5jb21lX3VrJywgICAgICBs"
    "OidVSyBQZW5zaW9uJywgIGc6J2luJ30sCiAge2s6J3RvdGFsX2luZmxvd3MnLCAgICAgICAgICBs"
    "OidUT1RBTCBJTicsICAgIGc6J2luJ30sCiAge2s6J2luY29tZV90YXgnLCAgICAgICAgICAgICBs"
    "OidJbmMgVGF4JywgICAgIGc6J291dCd9LAogIHtrOidtb3J0Z2FnZV9wYXltZW50JywgICAgICAg"
    "bDonTW9ydGdhZ2UnLCAgICBnOidvdXQnfSwKICB7azoncHJvcGVydHlfdGF4JywgICAgICAgICAg"
    "IGw6J1Byb3AgVGF4JywgICAgZzonb3V0J30sCiAge2s6J2hvbWVfaW5zdXJhbmNlJywgICAgICAg"
    "ICBsOidJbnN1cmFuY2UnLCAgIGc6J291dCd9LAogIHtrOidiYXNpY19zcGVuZGluZycsICAgICAg"
    "ICAgbDonQmFzaWMgU3BlbmQnLCBnOidvdXQnfSwKICB7azonY2hpbGRfY29zdHNfYXBwbGllZCcs"
    "ICAgIGw6J0NoaWxkIENvc3RzJywgZzonb3V0J30sCiAge2s6J21lZGljYWxfY29zdHNfYXBwbGll"
    "ZCcsICBsOidNZWRpY2FsJywgICAgIGc6J291dCd9LAogIHtrOidvbmVfb2ZmX2Nvc3RzX2FwcGxp"
    "ZWQnLCAgbDonT25lLU9mZnMnLCAgICBnOidvdXQnfSwKICB7azondG90YWxfb3V0Zmxvd3MnLCAg"
    "ICAgICAgIGw6J1RPVEFMIE9VVCcsICAgZzonb3V0J30sCiAge2s6J25ldF9jYXNoZmxvdycsICAg"
    "ICAgICAgICBsOidORVQgQ0FTSEZMT1cnLGc6J2NmJ30sCiAge2s6J2hvdXNlX3ZhbHVlJywgICAg"
    "ICAgICAgICBsOidIb3VzZSBWYWx1ZScsIGc6J3N1bSd9LAogIHtrOidtb3J0Z2FnZV9yZW1haW5p"
    "bmcnLCAgICAgbDonTW9ydGdhZ2UgUmVtJyxnOidzdW0nfSwKICB7azonaG91c2VfZXF1aXR5Jywg"
    "ICAgICAgICAgIGw6J0hvdXNlIEVxdWl0eScsZzonc3VtJ30sCiAge2s6J25ldF93b3J0aCcsICAg"
    "ICAgICAgICAgICBsOidORVQgV09SVEgnLCAgIGc6J3N1bSd9LAogIHtrOidyZWFsX25ldF93b3J0"
    "aCcsICAgICAgICAgbDonUmVhbCBOVyAoMjAyNiQpJyxnOidzdW0nfSwKXTsKCmNvbnN0IEdfTEFC"
    "RUwgPSB7aWQ6JycsIGludjonSU5WRVNUTUVOVFMnLCBpbjonSU5GTE9XUyArJywgb3V0OidPVVRG"
    "TE9XUyDiiJInLCBjZjonJywgc3VtOidTVU1NQVJZJ307CgovLyDilIDilIAgRk9STUFUIOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiBmbXQodikgewogIGlmICh2ID09"
    "PSAwKSByZXR1cm4gJ+KAlCc7CiAgY29uc3QgYWJzID0gTWF0aC5hYnModikudG9Mb2NhbGVTdHJp"
    "bmcoKTsKICByZXR1cm4gdiA8IDAgPyAnLSQnICsgYWJzIDogJyQnICsgYWJzOwp9CgovLyDilIDi"
    "lIAgU1RBVEUg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmxldCBzY2VuYXJp"
    "byA9ICdBJzsKCmZ1bmN0aW9uIHNldFNjZW5hcmlvKHMpIHsKICBzY2VuYXJpbyA9IHM7CiAgZG9j"
    "dW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1BJykuY2xhc3NOYW1lID0gJ3NjZW5hcmlvLWJ0biAn"
    "ICsgKHM9PT0nQScgPyAnYWN0aXZlJyA6ICdpbmFjdGl2ZScpOwogIGRvY3VtZW50LmdldEVsZW1l"
    "bnRCeUlkKCdidG4tQicpLmNsYXNzTmFtZSA9ICdzY2VuYXJpby1idG4gJyArIChzPT09J0InID8g"
    "J2FjdGl2ZScgOiAnaW5hY3RpdmUnKTsKICByZW5kZXIoKTsKfQoKLy8g4pSA4pSAIEJVSUxEIEdS"
    "T1VQIFNQQU5TIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgApmdW5jdGlvbiBidWlsZEdyb3VwcygpIHsKICBjb25zdCBvdXQgPSBbXTsgbGV0"
    "IGN1ciA9IG51bGwsIG4gPSAwOwogIENPTFMuZm9yRWFjaCgoYywgaSkgPT4gewogICAgaWYgKGMu"
    "ZyAhPT0gY3VyKSB7IGlmIChjdXIgIT09IG51bGwpIG91dC5wdXNoKHtnOiBjdXIsIG59KTsgY3Vy"
    "ID0gYy5nOyBuID0gMTsgfQogICAgZWxzZSBuKys7CiAgfSk7CiAgb3V0LnB1c2goe2c6IGN1ciwg"
    "bn0pOwogIHJldHVybiBvdXQ7Cn0KY29uc3QgR1JPVVBTID0gYnVpbGRHcm91cHMoKTsKCi8vIOKU"
    "gOKUgCBLUEkgUkVOREVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApmdW5jdGlvbiByZW5kZXJLUElz"
    "KHJvd3MpIHsKICBjb25zdCBsYXN0ID0gcm93c1tyb3dzLmxlbmd0aCAtIDFdOwogIGNvbnN0IHJl"
    "dCAgPSByb3dzLmZpbmQociA9PiByLmFnZSA9PT0gNjApIHx8IHJvd3NbMF07CiAgY29uc3QgY2Zn"
    "ID0gQ09ORklHW3NjZW5hcmlvXTsKICBjb25zdCBrcGlzID0gWwogICAge2w6J05ldCBXb3J0aCBh"
    "dCBSZXRpcmVtZW50JywgdjogcmV0Lm5ldF93b3J0aCwgICAgICAgICAgICBjOicjZmJiZjI0Jywg"
    "cGN0OmZhbHNlfSwKICAgIHtsOidOZXQgV29ydGggYXQgRGVhdGggKDg1KScsIHY6IGxhc3QubmV0"
    "X3dvcnRoLCAgICAgICAgICAgYzonI2ZiYmYyNCcsIHBjdDpmYWxzZX0sCiAgICB7bDonUG9ydGZv"
    "bGlvIGF0IERlYXRoJywgICAgICB2OiBsYXN0LmludmVzdG1lbnRfcG9ydGZvbGlvLGM6JyM2MGE1"
    "ZmEnLCBwY3Q6ZmFsc2V9LAogICAge2w6J0ludiBHYWluIFJhdGUnLCAgICAgICAgICAgdjogY2Zn"
    "Lmludl9nYWluX3JhdGUsICAgICAgICBjOicjNGFkZTgwJywgcGN0OnRydWV9LAogICAge2w6J1Rv"
    "dGFsIFRheCBQYWlkJywgICAgICAgICAgdjogcm93cy5yZWR1Y2UoKHMscik9PnMrci5pbmNvbWVf"
    "dGF4K3IuaW52ZXN0bWVudF90YXgsMCksIGM6JyNmODcxNzEnLCBwY3Q6ZmFsc2V9LAogICAge2w6"
    "J0luZmxhdGlvbiBSYXRlJywgICAgICAgICAgdjogY2ZnLmluZmxhdGlvbl9yYXRlLCAgICAgICBj"
    "OicjZmI5MjNjJywgcGN0OnRydWV9LAogIF07CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2tw"
    "aXMnKS5pbm5lckhUTUwgPSBrcGlzLm1hcChrID0+CiAgICBgPGRpdiBjbGFzcz0ia3BpIj4KICAg"
    "ICAgPGRpdiBjbGFzcz0ia3BpLWxhYmVsIj4ke2subC50b1VwcGVyQ2FzZSgpfTwvZGl2PgogICAg"
    "ICA8ZGl2IGNsYXNzPSJrcGktdmFsdWUiIHN0eWxlPSJjb2xvcjoke2suY30iPiR7ay5wY3QgPyBr"
    "LnYrJyUnIDogJyQnKyhrLnZ8fDApLnRvTG9jYWxlU3RyaW5nKCl9PC9kaXY+CiAgICA8L2Rpdj5g"
    "CiAgKS5qb2luKCcnKTsKfQoKLy8g4pSA4pSAIFRBQkxFIFJFTkRFUiDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAK"
    "ZnVuY3Rpb24gcmVuZGVyKCkgewogIGNvbnN0IHJvd3MgPSBEQVRBW3NjZW5hcmlvXTsKICByZW5k"
    "ZXJLUElzKHJvd3MpOwoKICAvLyB0aGVhZAogIGNvbnN0IGdoUm93ID0gR1JPVVBTLm1hcCgoe2cs"
    "IG59KSA9PgogICAgYDx0aCBjbGFzcz0iZ2ggZy0ke2d9IiBjb2xzcGFuPSIke259IiBzdHlsZT0i"
    "Ym9yZGVyLXJpZ2h0OjFweCBzb2xpZCB2YXIoLS1iKSI+JHtHX0xBQkVMW2ddfTwvdGg+YAogICku"
    "am9pbignJyk7CgogIGNvbnN0IGNoUm93ID0gQ09MUy5tYXAoYyA9PgogICAgYDx0aCBjbGFzcz0i"
    "Y2ggZy0ke2MuZ30iIHN0eWxlPSJib3JkZXItcmlnaHQ6MXB4IHNvbGlkIj4ke2MubH08L3RoPmAK"
    "ICApLmpvaW4oJycpOwoKICAvLyB0Ym9keQogIGNvbnN0IHRib2R5Um93cyA9IHJvd3MubWFwKChy"
    "b3csIHJpKSA9PiB7CiAgICBjb25zdCBwcmV2UGhhc2UgPSByaSA+IDAgPyByb3dzW3JpLTFdLnBo"
    "YXNlIDogcm93LnBoYXNlOwogICAgY29uc3QgcmV0Qm9yZGVyID0gcm93LnBoYXNlID09PSAnUmV0"
    "aXJlZCcgJiYgcHJldlBoYXNlID09PSAnV29ya2luZyc7CgogICAgY29uc3QgY2VsbHMgPSBDT0xT"
    "Lm1hcChjID0+IHsKICAgICAgY29uc3QgdiA9IHJvd1tjLmtdOwogICAgICBsZXQgY2xzID0gYGct"
    "JHtjLmd9YDsKICAgICAgbGV0IGV4dHJhID0gJyc7CgogICAgICBpZiAoYy5rID09PSAnbmV0X3dv"
    "cnRoJykgICAgICAgICAgZXh0cmEgPSAnIGMtbncnOwogICAgICBlbHNlIGlmIChjLmsgPT09ICdy"
    "ZWFsX25ldF93b3J0aCcpICBleHRyYSA9ICcgYy1udyc7CiAgICAgIGVsc2UgaWYgKGMuayA9PT0g"
    "J2ludmVzdG1lbnRfcG9ydGZvbGlvJykgZXh0cmEgPSAnIGMtcG9ydCc7CiAgICAgIGVsc2UgaWYg"
    "KGMuayA9PT0gJ3RvdGFsX2luZmxvd3MnKSBleHRyYSA9ICcgYy10aW4nOwogICAgICBlbHNlIGlm"
    "IChjLmsgPT09ICd0b3RhbF9vdXRmbG93cycpZXh0cmEgPSAnIGMtdG91dCc7CiAgICAgIGVsc2Ug"
    "aWYgKGMuayA9PT0gJ25ldF9jYXNoZmxvdycpICBleHRyYSA9IHYgPj0gMCA/ICcgYy1jZi1wJyA6"
    "ICcgYy1jZi1uJzsKICAgICAgZWxzZSBpZiAoYy5rID09PSAncGhhc2UnKSAgICAgICAgIGV4dHJh"
    "ID0gdiA9PT0gJ1JldGlyZWQnID8gJyBjLXJldCcgOiAnIGMtd29yayc7CgogICAgICBjb25zdCBp"
    "c051bSA9IHR5cGVvZiB2ID09PSAnbnVtYmVyJzsKICAgICAgY29uc3QgZGlzcGxheSA9ICFpc051"
    "bSA/IHYKICAgICAgICA6IChjLmsgPT09ICd5ZWFyJyB8fCBjLmsgPT09ICdhZ2UnKSA/IHYKICAg"
    "ICAgICA6IGZtdCh2KTsKCiAgICAgIHJldHVybiBgPHRkIGNsYXNzPSIke2Nsc30ke2V4dHJhfSAk"
    "e2lzTnVtICYmIGMuayAhPT0gJ3llYXInICYmIGMuayAhPT0gJ2FnZScgPyAnbnVtJyA6ICdsYmwn"
    "fSIKICAgICAgICAgICAgICAgICAgc3R5bGU9ImJvcmRlci1yaWdodDoxcHggc29saWQ7Ym9yZGVy"
    "LWJvdHRvbToxcHggc29saWQiPiR7ZGlzcGxheX08L3RkPmA7CiAgICB9KS5qb2luKCcnKTsKCiAg"
    "ICByZXR1cm4gYDx0ciBjbGFzcz0iJHtyZXRCb3JkZXIgPyAncmV0aXJlLWJvcmRlcicgOiAnJ30i"
    "CiAgICAgICAgICAgICAgIG9ubW91c2VlbnRlcj0idGhpcy5jbGFzc0xpc3QuYWRkKCdob3YnKSIK"
    "ICAgICAgICAgICAgICAgb25tb3VzZWxlYXZlPSJ0aGlzLmNsYXNzTGlzdC5yZW1vdmUoJ2hvdicp"
    "Ij4ke2NlbGxzfTwvdHI+YDsKICB9KS5qb2luKCcnKTsKCiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5"
    "SWQoJ3RibCcpLmlubmVySFRNTCA9CiAgICBgPHRoZWFkPjx0cj4ke2doUm93fTwvdHI+PHRyPiR7"
    "Y2hSb3d9PC90cj48L3RoZWFkPjx0Ym9keT4ke3Rib2R5Um93c308L3Rib2R5PmA7Cn0KCnJlbmRl"
    "cigpOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="
)

_html = _b64.b64decode(_TEMPLATE_B64).decode()
_html = _html.replace('__DATA_A__', _data_a).replace('__DATA_B__', _data_b).replace('"__CONFIG__"', _config).replace('__CONFIG__', _config)

_html_path = "financial_viewer.html"
with open(_html_path, 'w') as _f:
    _f.write(_html)
print(f"  HTML saved   -> {_html_path}")
