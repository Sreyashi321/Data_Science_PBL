"""
T20 World Cup 2026 - batting & bowling analysis
Datasets: Batting__1_.csv  (innings-level batting)
          MatchResults_refined___1_.xlsx (sheets: Batting, TeamBat, Bowling)

Run:  python t20_analysis.py
Needs: pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BAT_CSV = "/mnt/user-data/uploads/Batting__1_.csv"
XLSX = "/mnt/user-data/uploads/MatchResults_refined___1_.xlsx"
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ 1. CLEANING
TEAM_FIX = {
    "Netherland": "Netherlands", "Afganisthan": "Afghanistan", "Afganistan": "Afghanistan",
    "ENG": "England", "New zealand": "New Zealand", "Zimbabwae": "Zimbabwe",
    "oman": "Oman",
}

def clean_team(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.replace(TEAM_FIX)

def clean_player(s: pd.Series) -> pd.Series:
    # drop captain/keeper markers like (c), †, ?, * and extra spaces
    s = s.astype("string").str.replace(r"\(c\)", "", regex=True)
    s = s.str.replace(r"[†?*]", "", regex=True)
    # the raw data has a find/replace bug: "ind" became "India" inside names
    # (KamIndiau -> Kamindu, RavIndiara -> Ravindra)
    s = s.str.replace(r"(?<=[A-Za-z])India|India(?=[a-z])", "ind", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def overs_to_balls(o) -> float:
    """3.5 overs -> 23 balls (the decimal part is balls, not a fraction)."""
    try:
        whole, _, part = str(o).partition(".")
        return int(whole) * 6 + (int(part) if part else 0)
    except ValueError:
        return np.nan

def prep_batting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Team"], df["Vs"] = clean_team(df["Team"]), clean_team(df["Vs"])
    df["Player"] = clean_player(df["Player"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%y", errors="coerce")
    for c in ["R", "B", "M", "4s", "6s", "SR", "Chase", "Win"]:
        df[c] = to_num(df[c])
    df = df.dropna(subset=["Player", "R", "B"])
    df["NotOut"] = df["Dismissal"].astype(str).str.lower().str.contains("not out")
    df["BoundaryRuns"] = df["4s"] * 4 + df["6s"] * 6
    df["BoundaryPct"] = np.where(df["R"] > 0, df["BoundaryRuns"] / df["R"] * 100, 0)
    df["Match"] = df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Team"] + " v " + df["Vs"]
    return df

def prep_bowling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Team"], df["Vs"] = clean_team(df["Team"]), clean_team(df["Vs"])
    df["Bowler"] = clean_player(df["Bowler"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%y", errors="coerce")
    df = df.dropna(subset=["Bowler"])
    for c in ["M", "R", "W", "Dots", "4s", "6s", "Econ", "Wide", "NB"]:
        df[c] = to_num(df[c])
    df["Balls"] = df["O"].apply(overs_to_balls)
    return df

bat = prep_batting(pd.read_csv(BAT_CSV))
bowl = prep_bowling(pd.read_excel(XLSX, sheet_name="Bowling"))
team_bat = pd.read_excel(XLSX, sheet_name="TeamBat")  # has a 'Total' column

print("Batting rows:", len(bat), "| Bowling rows:", len(bowl))
print("Teams:", sorted(bat["Team"].dropna().unique()), "\n")

# ------------------------------------------------------------------ 2. BATTING
players = (bat.groupby(["Player", "Team"])
    .agg(Inns=("R", "size"), Runs=("R", "sum"), Balls=("B", "sum"),
         HS=("R", "max"), NotOuts=("NotOut", "sum"),
         Fours=("4s", "sum"), Sixes=("6s", "sum"),
         Fifties=("R", lambda x: ((x >= 50) & (x < 100)).sum()),
         Hundreds=("R", lambda x: (x >= 100).sum()))
    .reset_index())
dismissals = (players["Inns"] - players["NotOuts"]).replace(0, np.nan)
players["Avg"] = (players["Runs"] / dismissals).round(2)
players["SR"] = (players["Runs"] / players["Balls"] * 100).round(2)
players["BoundaryPct"] = ((players["Fours"] * 4 + players["Sixes"] * 6)
                          / players["Runs"] * 100).round(1)

print("=== Top 10 run scorers ===")
print(players.sort_values("Runs", ascending=False).head(10)
      [["Player", "Team", "Inns", "Runs", "HS", "Avg", "SR", "Fours", "Sixes"]]
      .to_string(index=False), "\n")

print("=== Top 10 strike rates (min 40 balls faced) ===")
print(players[players["Balls"] >= 40].sort_values("SR", ascending=False).head(10)
      [["Player", "Team", "Runs", "Balls", "SR"]].to_string(index=False), "\n")

print("=== Most sixes ===")
print(players.sort_values("Sixes", ascending=False).head(10)
      [["Player", "Team", "Sixes", "Fours"]].to_string(index=False), "\n")

# ------------------------------------------------------------------ 3. TEAM LEVEL
team_runs = (bat.groupby("Team")
    .agg(Runs=("R", "sum"), Balls=("B", "sum"), Sixes=("6s", "sum"), Fours=("4s", "sum"))
    .assign(SR=lambda d: (d["Runs"] / d["Balls"] * 100).round(2))
    .sort_values("Runs", ascending=False))
print("=== Team batting summary ===")
print(team_runs.to_string(), "\n")

# Chasing vs defending: share of innings that ended in a win
chase_win = (bat.drop_duplicates(["Date", "Team", "Vs"])
    .groupby("Chase")["Win"].agg(["mean", "count"]))
chase_win.index = chase_win.index.map({0: "Batting first", 1: "Chasing"})
chase_win["mean"] = (chase_win["mean"] * 100).round(1)
print("=== Win % : batting first vs chasing ===")
print(chase_win.rename(columns={"mean": "Win%", "count": "Innings"}), "\n")

# ------------------------------------------------------------------ 4. BOWLING
bowlers = (bowl.groupby(["Bowler", "Team"])
    .agg(Mat=("R", "size"), Balls=("Balls", "sum"), Runs=("R", "sum"),
         Wkts=("W", "sum"), Dots=("Dots", "sum"), Wides=("Wide", "sum"))
    .reset_index())
bowlers["Overs"] = (bowlers["Balls"] // 6 + (bowlers["Balls"] % 6) / 10)
bowlers["Econ"] = (bowlers["Runs"] / (bowlers["Balls"] / 6)).round(2)
bowlers["Avg"] = (bowlers["Runs"] / bowlers["Wkts"].replace(0, np.nan)).round(2)
bowlers["DotPct"] = (bowlers["Dots"] / bowlers["Balls"] * 100).round(1)

print("=== Top 10 wicket takers ===")
print(bowlers.sort_values(["Wkts", "Econ"], ascending=[False, True]).head(10)
      [["Bowler", "Team", "Mat", "Overs", "Wkts", "Econ", "Avg"]]
      .to_string(index=False), "\n")

print("=== Best economy (min 10 overs) ===")
print(bowlers[bowlers["Balls"] >= 60].sort_values("Econ").head(10)
      [["Bowler", "Team", "Overs", "Econ", "DotPct"]].to_string(index=False), "\n")

# ------------------------------------------------------------------ 5. PLOTS
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

top = players.sort_values("Runs", ascending=False).head(10)
sns.barplot(data=top, y="Player", x="Runs", hue="Team", dodge=False, ax=ax[0, 0])
ax[0, 0].set_title("Top 10 run scorers")
ax[0, 0].legend(fontsize=7)

q = players[players["Balls"] >= 40]
sns.scatterplot(data=q, x="SR", y="Avg", size="Runs", hue="Team",
                legend=False, ax=ax[0, 1], sizes=(20, 250))
ax[0, 1].set_title("Strike rate vs average (min 40 balls)")

sns.barplot(data=team_runs.reset_index(), x="Sixes", y="Team", ax=ax[1, 0],
            color="tab:orange")
ax[1, 0].set_title("Sixes hit by team")

top_b = bowlers.sort_values(["Wkts", "Econ"], ascending=[False, True]).head(10)
sns.barplot(data=top_b, y="Bowler", x="Wkts", color="tab:green", ax=ax[1, 1])
ax[1, 1].set_title("Top 10 wicket takers")

plt.tight_layout()
plt.savefig("t20_overview.png", dpi=150)
plt.show()
print("Saved t20_overview.png\n")

# ------------------------------------------------------------------ 6. SIMPLE ML MODEL
# Predict whether a team won using its batting-innings features.
inn = (bat.groupby(["Date", "Team", "Vs"])
    .agg(Runs=("R", "sum"), Balls=("B", "sum"), Fours=("4s", "sum"),
         Sixes=("6s", "sum"), Chase=("Chase", "first"), Win=("Win", "first"))
    .reset_index())
inn["SR"] = inn["Runs"] / inn["Balls"] * 100
inn["BoundaryPct"] = (inn["Fours"] * 4 + inn["Sixes"] * 6) / inn["Runs"] * 100

features = ["Runs", "SR", "Fours", "Sixes", "BoundaryPct", "Chase"]
X, y = inn[features].fillna(0), inn["Win"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_tr, y_tr)
pred = model.predict(X_te)
print("=== Win prediction (Random Forest) ===")
print("Accuracy:", round(accuracy_score(y_te, pred), 3))
print(classification_report(y_te, pred))
print("Feature importance:")
print(pd.Series(model.feature_importances_, index=features)
      .sort_values(ascending=False).round(3))