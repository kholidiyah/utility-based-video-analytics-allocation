import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # penting untuk running dari terminal/WSL tanpa GUI
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# BASE PATHS
# ============================================================
BASE_DIR = Path("/mnt/c/Data/2025/Bimbingan/Semester_7/Joint_Optimization")

file_acc_util = BASE_DIR / "Summary_Hasil_Simulasi_Accuracy_vs_Utility.xlsx"
file_br_acc   = BASE_DIR / "Summary_Hasil_Simulasi_Bitrate_vs_Accuracy.xlsx"
file_br_util  = BASE_DIR / "Summary_Hasil_Simulasi_Bitrate_vs_Utility.xlsx"

OUT_DIR = BASE_DIR / "figures_regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Practical threshold
B_PRACTICAL = 20000
BITRATE_UNIT = "bps"


# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "axes.linewidth": 1.1,
})


# ============================================================
# HELPER: CLEAN EXCEL FORMAT
# ============================================================
def clean_two_row_excel(path, x_name_left, y_name_right):
    """
    Reads the Excel summary files with the following format:
    row 0 = subheader, e.g., Bitrate / Accuracy / Utility
    row 1.. = data

    Output:
    long-format dataframe with columns:
    Model, Stream, Allocator, x_name_left, y_name_right
    """
    raw = pd.read_excel(path)
    raw = raw.dropna(how="all").reset_index(drop=True)

    # first row contains subheaders
    subheader = raw.iloc[0].fillna("").astype(str).str.strip().tolist()

    # data starts from second row
    df = raw.iloc[1:].copy().reset_index(drop=True)

    df.columns = [
        "Model",
        "Stream",
        f"Alpha_{subheader[2]}",
        f"Alpha_{subheader[3]}",
        f"Markov_{subheader[4]}",
        f"Markov_{subheader[5]}",
        f"Lyapunov_{subheader[6]}",
        f"Lyapunov_{subheader[7]}",
    ]

    # fix typo from Excel header
    df.columns = [c.replace("Accuray", "Accuracy") for c in df.columns]

    # forward-fill model names because Excel merged cells appear as NaN
    df["Model"] = df["Model"].ffill()

    # convert numeric columns
    for c in df.columns:
        if c != "Model":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # remove rows without stream id
    df = df.dropna(subset=["Stream"]).copy()

    rows = []
    allocators = ["Alpha", "Markov", "Lyapunov"]

    for _, row in df.iterrows():
        for alloc in allocators:
            x_col = f"{alloc}_{subheader[2]}".replace("Accuray", "Accuracy")
            y_col = f"{alloc}_{subheader[3]}".replace("Accuray", "Accuracy")

            rows.append({
                "Model": row["Model"],
                "Stream": int(row["Stream"]),
                "Allocator": alloc,
                x_name_left: row[x_col],
                y_name_right: row[y_col],
            })

    long_df = pd.DataFrame(rows)
    long_df = long_df.dropna(subset=[x_name_left, y_name_right]).reset_index(drop=True)
    return long_df


# ============================================================
# LOAD DATA
# ============================================================
df_acc_util = clean_two_row_excel(file_acc_util, "Accuracy", "Utility")
df_br_acc   = clean_two_row_excel(file_br_acc, "Bitrate", "Accuracy")
df_br_util  = clean_two_row_excel(file_br_util, "Bitrate", "Utility")


# ============================================================
# PRINT CHECK
# ============================================================
print("\nAccuracy vs Utility")
print(df_acc_util.head().to_string(index=False))

print(f"\nBitrate vs Accuracy ({BITRATE_UNIT})")
print(df_br_acc.head().to_string(index=False))

print(f"\nBitrate vs Utility ({BITRATE_UNIT})")
print(df_br_util.head().to_string(index=False))


# ============================================================
# PRACTICAL FILTER
# ============================================================
df_br_acc_practical  = df_br_acc[df_br_acc["Bitrate"] <= B_PRACTICAL].copy()
df_br_util_practical = df_br_util[df_br_util["Bitrate"] <= B_PRACTICAL].copy()

valid_keys = df_br_acc_practical[["Model", "Stream", "Allocator"]].drop_duplicates()

df_acc_util_practical = df_acc_util.merge(
    valid_keys,
    on=["Model", "Stream", "Allocator"],
    how="inner"
)


# ============================================================
# STYLE MAP
# ============================================================
allocator_order = ["Alpha", "Markov", "Lyapunov"]
model_order = ["YOLOv5", "YOLOv8", "YOLOv10", "YOLOv11"]

allocator_label = {
    "Alpha": "Alpha Fairness",
    "Markov": "Markov Chain",
    "Lyapunov": "Lyapunov",
}

color_map = {
    "Alpha": "tab:blue",
    "Markov": "tab:orange",
    "Lyapunov": "tab:green",
}

marker_map = {
    "YOLOv5": "o",
    "YOLOv8": "s",
    "YOLOv10": "^",
    "YOLOv11": "D",
}


# ============================================================
# LEGEND HELPERS
# ============================================================
def add_dual_legend(ax, loc_color="lower right"):
    allocator_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=allocator_label[a],
            markerfacecolor=color_map[a],
            markeredgecolor="black",
            markersize=9,
        )
        for a in allocator_order
    ]

    model_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[m],
            color="black",
            label=m,
            linestyle="None",
            markersize=9,
        )
        for m in model_order
    ]

    leg1 = ax.legend(
        handles=allocator_handles,
        title="Allocator",
        loc=loc_color,
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=model_handles,
        title="YOLO Model",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )


def annotate_streams(ax, df, xcol, ycol, logx=False):
    for _, row in df.iterrows():
        x = row[xcol]
        y = row[ycol]

        if pd.isna(x) or pd.isna(y):
            continue

        if logx:
            x_text = x * 1.06
        else:
            x_range = df[xcol].max() - df[xcol].min()
            x_text = x + 0.01 * x_range if x_range > 0 else x

        y_range = df[ycol].max() - df[ycol].min()
        y_text = y + 0.015 * y_range if y_range > 0 else y

        ax.text(
            x_text,
            y_text,
            f"S{int(row['Stream'])}",
            fontsize=10,
            ha="left",
            va="bottom",
        )


# ============================================================
# PLOT FUNCTION
# ============================================================
def scatter_plot(
    df,
    xcol,
    ycol,
    title,
    xlabel,
    ylabel,
    logx=False,
    save_stem=None,
    annotate=True,
):
    fig, ax = plt.subplots(figsize=(11, 7.2))

    for allocator in allocator_order:
        for model in model_order:
            sub = df[(df["Allocator"] == allocator) & (df["Model"] == model)]
            if sub.empty:
                continue

            ax.scatter(
                sub[xcol],
                sub[ycol],
                color=color_map[allocator],
                marker=marker_map.get(model, "o"),
                s=120,
                alpha=0.86,
                edgecolors="black",
                linewidths=0.5,
            )

    if logx:
        ax.set_xscale("log")

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)

    if annotate:
        annotate_streams(ax, df, xcol, ycol, logx=logx)

    add_dual_legend(ax)

    fig.tight_layout()

    if save_stem is not None:
        png_path = OUT_DIR / f"{save_stem}.png"
        pdf_path = OUT_DIR / f"{save_stem}.pdf"

        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    plt.close(fig)


# ============================================================
# COMBINED VERTICAL FIGURE
# ============================================================
def combined_three_panel(
    df1, df2, df3,
    title1, title2, title3,
    save_stem,
    regime_title,
    log_bitrate=True,
):
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 20))

    plot_specs = [
        (axes[0], df1, "Accuracy", "Utility", title1, "Accuracy", "Utility", False),
        (axes[1], df2, "Bitrate", "Accuracy", title2, f"Bitrate ({BITRATE_UNIT})", "Accuracy", log_bitrate),
        (axes[2], df3, "Bitrate", "Utility", title3, f"Bitrate ({BITRATE_UNIT})", "Utility", log_bitrate),
    ]

    for ax, df, xcol, ycol, title, xlabel, ylabel, logx in plot_specs:
        for allocator in allocator_order:
            for model in model_order:
                sub = df[(df["Allocator"] == allocator) & (df["Model"] == model)]
                if sub.empty:
                    continue

                ax.scatter(
                    sub[xcol],
                    sub[ycol],
                    color=color_map[allocator],
                    marker=marker_map.get(model, "o"),
                    s=115,
                    alpha=0.86,
                    edgecolors="black",
                    linewidths=0.5,
                )

        if logx:
            ax.set_xscale("log")

        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        annotate_streams(ax, df, xcol, ycol, logx=logx)

    # shared legends at bottom
    allocator_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=allocator_label[a],
            markerfacecolor=color_map[a],
            markeredgecolor="black",
            markersize=10,
        )
        for a in allocator_order
    ]

    model_handles = [
        Line2D(
            [0], [0],
            marker=marker_map[m],
            color="black",
            label=m,
            linestyle="None",
            markersize=10,
        )
        for m in model_order
    ]

    fig.legend(
        handles=allocator_handles,
        title="Resource Allocator",
        loc="lower center",
        bbox_to_anchor=(0.35, 0.01),
        ncol=3,
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )

    fig.legend(
        handles=model_handles,
        title="YOLO Model",
        loc="lower center",
        bbox_to_anchor=(0.72, 0.01),
        ncol=4,
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )

    fig.suptitle(regime_title, y=0.995, fontsize=18)
    fig.tight_layout(rect=[0, 0.055, 1, 0.985])

    png_path = OUT_DIR / f"{save_stem}.png"
    pdf_path = OUT_DIR / f"{save_stem}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


# ============================================================
# CREATE INDIVIDUAL FULL-REGIME PLOTS
# ============================================================
scatter_plot(
    df_acc_util,
    xcol="Accuracy",
    ycol="Utility",
    title="Full-Regime: Accuracy vs Utility",
    xlabel="Accuracy",
    ylabel="Utility",
    logx=False,
    save_stem="full_regime_accuracy_vs_utility",
)

scatter_plot(
    df_br_acc,
    xcol="Bitrate",
    ycol="Accuracy",
    title="Full-Regime: Bitrate vs Accuracy",
    xlabel=f"Bitrate ({BITRATE_UNIT})",
    ylabel="Accuracy",
    logx=True,
    save_stem="full_regime_bitrate_vs_accuracy",
)

scatter_plot(
    df_br_util,
    xcol="Bitrate",
    ycol="Utility",
    title="Full-Regime: Bitrate vs Utility",
    xlabel=f"Bitrate ({BITRATE_UNIT})",
    ylabel="Utility",
    logx=True,
    save_stem="full_regime_bitrate_vs_utility",
)


# ============================================================
# CREATE INDIVIDUAL PRACTICAL-REGIME PLOTS
# ============================================================
scatter_plot(
    df_acc_util_practical,
    xcol="Accuracy",
    ycol="Utility",
    title="Practical-Regime: Accuracy vs Utility",
    xlabel="Accuracy",
    ylabel="Utility",
    logx=False,
    save_stem="practical_regime_accuracy_vs_utility",
)

scatter_plot(
    df_br_acc_practical,
    xcol="Bitrate",
    ycol="Accuracy",
    title="Practical-Regime: Bitrate vs Accuracy",
    xlabel=f"Bitrate ({BITRATE_UNIT})",
    ylabel="Accuracy",
    logx=False,
    save_stem="practical_regime_bitrate_vs_accuracy",
)

scatter_plot(
    df_br_util_practical,
    xcol="Bitrate",
    ycol="Utility",
    title="Practical-Regime: Bitrate vs Utility",
    xlabel=f"Bitrate ({BITRATE_UNIT})",
    ylabel="Utility",
    logx=False,
    save_stem="practical_regime_bitrate_vs_utility",
)


# ============================================================
# CREATE COMBINED VERTICAL FIGURES
# ============================================================
combined_three_panel(
    df_acc_util,
    df_br_acc,
    df_br_util,
    title1="(a) Accuracy--Utility",
    title2="(b) Bitrate--Accuracy",
    title3="(c) Bitrate--Utility",
    save_stem="combined_full_regime_three_panel_vertical",
    regime_title="Full-Regime Analysis",
    log_bitrate=True,
)

combined_three_panel(
    df_acc_util_practical,
    df_br_acc_practical,
    df_br_util_practical,
    title1="(a) Accuracy--Utility",
    title2="(b) Bitrate--Accuracy",
    title3="(c) Bitrate--Utility",
    save_stem="combined_practical_regime_three_panel_vertical",
    regime_title=f"Practical-Regime Analysis, B_practical = {B_PRACTICAL} {BITRATE_UNIT}",
    log_bitrate=False,
)


# ============================================================
# SUMMARY COUNTS
# ============================================================
summary_counts = (
    df_br_acc.assign(
        Regime=np.where(
            df_br_acc["Bitrate"] <= B_PRACTICAL,
            "Practical",
            "Non-Practical",
        )
    )
    .groupby(["Allocator", "Regime"])
    .size()
    .unstack(fill_value=0)
)

# Pastikan urutan kolom konsisten
for col in ["Practical", "Non-Practical"]:
    if col not in summary_counts.columns:
        summary_counts[col] = 0

summary_counts = summary_counts[["Practical", "Non-Practical"]]
summary_counts["Total"] = summary_counts["Practical"] + summary_counts["Non-Practical"]
summary_counts["Threshold"] = f"{B_PRACTICAL} {BITRATE_UNIT}"

summary_counts = summary_counts.reindex(allocator_order)

print(f"\nSummary counts per allocator with B_PRACTICAL = {B_PRACTICAL} {BITRATE_UNIT}:")
print(summary_counts.to_string())

summary_path = OUT_DIR / "operating_regime_classification.csv"
summary_counts.to_csv(summary_path)
print(f"\nSaved summary: {summary_path}")


# ============================================================
# QUANTITATIVE SUMMARY TABLE
# ============================================================

# Gabungkan data Accuracy-Utility dengan Bitrate
df_quant = df_acc_util.merge(
    df_br_acc[["Model", "Stream", "Allocator", "Bitrate"]],
    on=["Model", "Stream", "Allocator"],
    how="left"
)

summary_quant = (
    df_quant
    .groupby("Allocator")
    .agg(
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_Reported_Utility=("Utility", "mean"),
        Mean_Bitrate=("Bitrate", "mean"),
        Sum_Utility=("Utility", "sum"),
        Sum_Bitrate=("Bitrate", "sum"),
        Total=("Bitrate", "count"),
        Practical=("Bitrate", lambda x: (x <= B_PRACTICAL).sum()),
    )
    .reset_index()
)

summary_quant["Bitrate_Efficiency"] = (
    summary_quant["Sum_Utility"] / summary_quant["Sum_Bitrate"]
)

summary_quant["Practical_Total"] = (
    summary_quant["Practical"].astype(str) + "/" + summary_quant["Total"].astype(str)
)

# Reorder allocator rows
summary_quant["Allocator"] = pd.Categorical(
    summary_quant["Allocator"],
    categories=allocator_order,
    ordered=True,
)
summary_quant = summary_quant.sort_values("Allocator").reset_index(drop=True)

summary_quant_final = summary_quant[
    [
        "Allocator",
        "Mean_Accuracy",
        "Mean_Reported_Utility",
        "Mean_Bitrate",
        "Bitrate_Efficiency",
        "Practical_Total",
    ]
].copy()

summary_quant_final = summary_quant_final.rename(columns={
    "Mean_Bitrate": f"Mean_Bitrate_{BITRATE_UNIT}",
    "Bitrate_Efficiency": f"Bitrate_Efficiency_utility_per_{BITRATE_UNIT}",
})

print("\nQuantitative Summary:")
print(summary_quant_final.to_string(index=False))

summary_quant_path = OUT_DIR / "quantitative_summary_allocation_performance.csv"
summary_quant_final.to_csv(summary_quant_path, index=False)
print(f"\nSaved quantitative summary: {summary_quant_path}")


# ============================================================
# SAVE LATEX TABLE FOR PAPER
# ============================================================
latex_table_path = OUT_DIR / "quantitative_summary_latex_table.tex"

with open(latex_table_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table*}[!htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Quantitative Summary of Allocation Performance}\n")
    f.write("\\label{tab:quantitative-summary}\n")
    f.write("\\renewcommand{\\arraystretch}{1.15}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{lccccc}\n")
    f.write("\\hline\n")
    f.write("\\textbf{Allocator} &\n")
    f.write("\\textbf{Mean Accuracy} &\n")
    f.write("\\textbf{Mean Reported Utility} &\n")
    f.write(f"\\textbf{{Mean Bitrate ({BITRATE_UNIT})}} &\n")
    f.write(f"\\textbf{{Bitrate Efficiency (utility/{BITRATE_UNIT})}} &\n")
    f.write("\\textbf{Practical/Total} \\\\\n")
    f.write("\\hline\n")

    for _, row in summary_quant_final.iterrows():
        allocator_name = allocator_label.get(str(row["Allocator"]), str(row["Allocator"]))
        mean_acc = float(row["Mean_Accuracy"])
        mean_util = float(row["Mean_Reported_Utility"])
        mean_bitrate = float(row[f"Mean_Bitrate_{BITRATE_UNIT}"])
        efficiency = float(row[f"Bitrate_Efficiency_utility_per_{BITRATE_UNIT}"])
        practical_total = row["Practical_Total"]

        f.write(
            f"{allocator_name} & "
            f"{mean_acc:.4f} & "
            f"{mean_util:.4f} & "
            f"{mean_bitrate:.2f} & "
            f"\\({efficiency:.3e}\\) & "
            f"{practical_total} \\\\\n"
        )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table*}\n")

print(f"Saved LaTeX table: {latex_table_path}")


# ============================================================
# SAVE CLEANED LONG-FORM DATA
# ============================================================
df_acc_util.to_csv(OUT_DIR / "clean_accuracy_vs_utility_long.csv", index=False)
df_br_acc.to_csv(OUT_DIR / "clean_bitrate_vs_accuracy_long.csv", index=False)
df_br_util.to_csv(OUT_DIR / "clean_bitrate_vs_utility_long.csv", index=False)

print("\nAll figures and cleaned data have been saved in:")
print(OUT_DIR)