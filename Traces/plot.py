import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0) Files
# =========================================================
FILE_STM32_SBOX    = "index_time_plaintext_stm32.csv"
FILE_RP2350_SBOX   = "index_time_plaintext_rp2350_sbox.csv"
FILE_RP2350_TTABLE = "index_time_plaintext_rp2350_ttable.csv"

OUTPUT_DIR = "."
OUT_PDF = os.path.join(OUTPUT_DIR, "distribution_stability_2x3.pdf")

# =========================================================
# 1) Unified style
# =========================================================
PALETTE = {
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}

REDUCE_SIZE = 2
FIGSIZE_2COL_2x3 = (7.1, 4.8)
MAX_TRACES_FOR_OVERVIEW = 100000
STABILITY_PLOT_POINTS = 20000

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def apply_paper_style(reduce_size: int = 2):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 9 - reduce_size,
        "axes.labelsize": 9 - reduce_size,
        "axes.titlesize": 9 - reduce_size,
        "xtick.labelsize": 8 - reduce_size,
        "ytick.labelsize": 8 - reduce_size,
        "legend.fontsize": 8 - reduce_size,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.45,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })

def iqr_clean(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    q1 = df["Cycles"].quantile(0.25)
    q3 = df["Cycles"].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return df[(df["Cycles"] >= lo) & (df["Cycles"] <= hi)].copy()

def uniform_sample_df(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    n = len(df)
    if n <= target_n:
        return df
    idx = np.linspace(0, n - 1, target_n, dtype=int)
    return df.iloc[idx].copy()

def pick_x_index_col(df: pd.DataFrame) -> str:
    if "Index" in df.columns:
        return "Index"
    if "Counter" in df.columns:
        return "Counter"
    df["__row__"] = np.arange(len(df), dtype=int)
    return "__row__"

def load_clean_for_overview(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    df = df_raw.head(MAX_TRACES_FOR_OVERVIEW).copy()
    df = iqr_clean(df, k=3.0)
    return df

# =========================================================
# 2) Plot helpers
# =========================================================
def plot_distribution(ax, cycles: np.ndarray):
    sns.histplot(
        cycles, bins=80, kde=True,
        color=PALETTE["sky"],
        edgecolor="white", linewidth=0.4,
        ax=ax
    )
    ax.set_xlabel("Clock Cycles")
    ax.set_ylabel("Frequency")

def plot_stability(ax, x_vals: np.ndarray, cycles: np.ndarray):
    ax.scatter(
        x_vals, cycles,
        s=2.0, c=PALETTE["purple"], alpha=0.45, edgecolors="none"
    )
    ax.set_xlabel("Trace Index (Chronological Order)")
    ax.set_ylabel("Clock Cycles")

    y_min, y_max = float(np.min(cycles)), float(np.max(cycles))
    margin = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
    ax.set_ylim(y_min - margin, y_max + margin)

# =========================================================
# 3) Main
# =========================================================
if __name__ == "__main__":
    ensure_outdir(OUTPUT_DIR)
    apply_paper_style(REDUCE_SIZE)

    datasets = [
        ("STM32 AES (S-box)",    FILE_STM32_SBOX),
        ("RP2350 AES (S-box)",   FILE_RP2350_SBOX),
        ("RP2350 AES (T-table)", FILE_RP2350_TTABLE),
    ]

    dfs = []
    for title, path in datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = load_clean_for_overview(path)
        dfs.append((title, df))


    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_2COL_2x3, sharey=False)

    # --- Row 1: distribution ---
    for col, (title, df) in enumerate(dfs):
        ax = axes[0, col]
        plot_distribution(ax, df["Cycles"].to_numpy())
        ax.set_title(title, pad=4)


        if col != 0:
            ax.set_ylabel("")

    # --- Row 2: stability ---
    for col, (title, df) in enumerate(dfs):
        ax = axes[1, col]
        df_plot = uniform_sample_df(df, STABILITY_PLOT_POINTS)
        xcol = pick_x_index_col(df_plot)

        plot_stability(ax, df_plot[xcol].to_numpy(), df_plot["Cycles"].to_numpy())


        if col != 0:
            ax.set_ylabel("")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.32, top=0.90)
    #fig.suptitle("Cycle Distribution and Stability", y=0.98)

    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PDF}")
