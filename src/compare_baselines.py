"""
대조군 A (컨센서스) vs 대조군 B (골든크로스) 비교 차트
results/baseline_comparison.png
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from utils import TICKERS, get_baseline_dir, get_latest_baseline_dir

LATEST_DIR   = get_latest_baseline_dir()
BASELINE_DIR = get_baseline_dir()

# ── 한글 폰트 설정 ────────────────────────────────────────
def set_korean_font():
    candidates = [
        "Malgun Gothic", "맑은 고딕",
        "NanumGothic", "NanumBarunGothic",
        "AppleGothic", "Gulim",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 데이터 로드 ───────────────────────────────────────────
cons  = pd.read_csv(os.path.join(LATEST_DIR, "consensus_returns.csv"))
gold  = pd.read_csv(os.path.join(LATEST_DIR, "golden_returns.csv"))

# 종목명 역매핑 (ticker → name)
ticker2name = {v: k for k, v in TICKERS.items()}

# ── 종목별 Hit Rate 계산 ──────────────────────────────────
def hit_rate_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("ticker")["return_pct"].agg(
        hit_rate=lambda x: (x > 0).mean() * 100,
        n="count",
    ).reset_index()
    grp["name"] = grp["ticker"].map(ticker2name)
    return grp

cons_hr = hit_rate_by_ticker(cons)
gold_hr = hit_rate_by_ticker(gold)

# 공통 종목 기준 정렬 (컨센서스 hit_rate 내림차순)
merged = pd.merge(
    cons_hr[["ticker", "name", "hit_rate"]].rename(columns={"hit_rate": "cons_hr"}),
    gold_hr[["ticker", "hit_rate"]].rename(columns={"hit_rate": "gold_hr"}),
    on="ticker", how="outer",
).fillna(0).sort_values("cons_hr", ascending=False)

# ── Sharpe 계산 ───────────────────────────────────────────
def sharpe(series: pd.Series, rf: float = 0.0) -> float:
    excess = series - rf
    return excess.mean() / excess.std() if excess.std() != 0 else 0.0

cons_sharpe = sharpe(cons["return_pct"])
gold_sharpe = sharpe(gold["return_pct"])

cons_hit  = (cons["return_pct"] > 0).mean() * 100
gold_hit  = (gold["return_pct"] > 0).mean() * 100
cons_mean = cons["return_pct"].mean()
gold_mean = gold["return_pct"].mean()

# ── 색상 ──────────────────────────────────────────────────
C_CONS = "#2196F3"   # 파랑 - 컨센서스
C_GOLD = "#FF9800"   # 주황 - 골든크로스

# ── Figure 레이아웃 ───────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor="#F8F9FA")
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.08)

ax1 = fig.add_subplot(gs[0, :])   # 상단 전체: 종목별 Hit Rate
ax2 = fig.add_subplot(gs[1, 0])   # 하단 좌: 수익률 분포
ax3 = fig.add_subplot(gs[1, 1])   # 하단 우: 요약 테이블

fig.suptitle("대조군 A (컨센서스) vs 대조군 B (골든크로스)  |  2025-07-01 ~ 2026-03-12",
             fontsize=15, fontweight="bold", y=0.97)

# ──────────────────────────────────────────────────────────
# ① 종목별 Hit Rate 비교 막대그래프
# ──────────────────────────────────────────────────────────
n      = len(merged)
x      = np.arange(n)
width  = 0.38

bars1 = ax1.bar(x - width/2, merged["cons_hr"], width,
                label="대조군 A  컨센서스", color=C_CONS, alpha=0.88, zorder=3)
bars2 = ax1.bar(x + width/2, merged["gold_hr"], width,
                label="대조군 B  골든크로스", color=C_GOLD, alpha=0.88, zorder=3)

ax1.axhline(50, color="gray", linewidth=0.9, linestyle="--", zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(merged["name"].fillna(merged["ticker"]), rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("Hit Rate (%)")
ax1.set_ylim(0, 105)
ax1.set_title("① 종목별 Hit Rate 비교", fontsize=12, fontweight="bold", pad=8)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.4, zorder=1)
ax1.set_facecolor("white")

# 값 레이블
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color=C_CONS, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color="#E65100", fontweight="bold")

# ──────────────────────────────────────────────────────────
# ② 수익률 분포 히스토그램
# ──────────────────────────────────────────────────────────
bins = np.linspace(
    min(cons["return_pct"].min(), gold["return_pct"].min()) - 2,
    max(cons["return_pct"].max(), gold["return_pct"].max()) + 2,
    40,
)

ax2.hist(cons["return_pct"], bins=bins, alpha=0.60, color=C_CONS,
         label=f"컨센서스  (n={len(cons)})", edgecolor="white", linewidth=0.4, zorder=3)
ax2.hist(gold["return_pct"], bins=bins, alpha=0.55, color=C_GOLD,
         label=f"골든크로스  (n={len(gold)})", edgecolor="white", linewidth=0.4, zorder=2)

ax2.axvline(cons_mean, color=C_CONS, linewidth=1.8, linestyle="--", zorder=4,
            label=f"컨센서스 평균 {cons_mean:+.2f}%")
ax2.axvline(gold_mean, color=C_GOLD, linewidth=1.8, linestyle="--", zorder=4,
            label=f"골든크로스 평균 {gold_mean:+.2f}%")
ax2.axvline(0, color="black", linewidth=1.0, linestyle="-", zorder=5)

ax2.set_xlabel("수익률 (%)")
ax2.set_ylabel("빈도")
ax2.set_title("② 전체 수익률 분포", fontsize=12, fontweight="bold", pad=8)
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.3, zorder=1)
ax2.set_facecolor("white")

# ──────────────────────────────────────────────────────────
# ③ 요약 테이블
# ──────────────────────────────────────────────────────────
ax3.axis("off")
ax3.set_title("③ 전략 요약 비교", fontsize=12, fontweight="bold", pad=8)

col_labels = ["지표", "대조군 A\n컨센서스", "대조군 B\n골든크로스"]
rows = [
    ["신호 수",         f"{len(cons):,}",               f"{len(gold):,}"],
    ["Hit Rate",        f"{cons_hit:.1f}%",              f"{gold_hit:.1f}%"],
    ["평균 수익률",     f"{cons_mean:+.2f}%",            f"{gold_mean:+.2f}%"],
    ["중앙값 수익률",   f"{cons['return_pct'].median():+.2f}%",
                        f"{gold['return_pct'].median():+.2f}%"],
    ["표준편차",        f"{cons['return_pct'].std():.2f}%",
                        f"{gold['return_pct'].std():.2f}%"],
    ["Sharpe",          f"{cons_sharpe:.3f}",            f"{gold_sharpe:.3f}"],
    ["최대 수익",       f"{cons['return_pct'].max():+.1f}%",
                        f"{gold['return_pct'].max():+.1f}%"],
    ["최대 손실",       f"{cons['return_pct'].min():+.1f}%",
                        f"{gold['return_pct'].min():+.1f}%"],
]

table = ax3.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.0, 2.1)

# 헤더 색상
for j in range(3):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")

# 행 교차 색상 + 전략 컬럼 하이라이트
for i in range(1, len(rows) + 1):
    bg = "#F5F5F5" if i % 2 == 0 else "white"
    table[i, 0].set_facecolor(bg)
    table[i, 1].set_facecolor("#E3F2FD")   # 연파랑 - 컨센서스
    table[i, 2].set_facecolor("#FFF3E0")   # 연주황 - 골든크로스

# Hit Rate 셀 강조
hit_row = 2
better_col = 1 if cons_hit >= gold_hit else 2
table[hit_row, better_col].set_facecolor("#A5D6A7")   # 초록 - 우수

# 평균 수익률 강조
ret_row = 3
better_col2 = 1 if cons_mean >= gold_mean else 2
table[ret_row, better_col2].set_facecolor("#A5D6A7")

# Sharpe 강조
sh_row = 6
better_col3 = 1 if cons_sharpe >= gold_sharpe else 2
table[sh_row, better_col3].set_facecolor("#A5D6A7")

# ── 저장 ──────────────────────────────────────────────────
out = os.path.join(BASELINE_DIR, "baseline_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
shutil.copy(out, os.path.join(LATEST_DIR, "baseline_comparison.png"))
print(f"저장 완료: {out}")


if __name__ == "__main__":
    pass
