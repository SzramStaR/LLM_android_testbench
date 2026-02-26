import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict
import math


import matplotlib
import numpy as np
import pandas as pd
from aquarel import load_theme
from matplotlib import colormaps, pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Patch, Rectangle
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


from utils import DraggableText

matplotlib.set_loglevel("error")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]


THEME_NAME = "scientific"


def _save_markdown_table_for_plot(
    save_path: Optional[str], datasets: list[dict], table_columns: list[tuple[str, str]]
):
    """Save a markdown table with exactly the points plotted"""
    if not save_path:
        return
    try:
        out_path = os.path.splitext(save_path)[0] + ".data.md"
        frames = []
        for data in datasets or []:
            df = data.get("df")
            if df is None or getattr(df, "empty", True):
                continue
            cols = [c for c, _ in table_columns if c in df.columns]
            if not cols:
                continue
            subset = df[cols].dropna(subset=cols)
            frames.append(subset)
        if not frames:
            return
        combined = pd.concat(frames, axis=0, ignore_index=True)
        ordered_cols = [c for c, _ in table_columns if c in combined.columns]
        combined = combined[ordered_cols]

        headers = [h for _, h in table_columns if _ in ordered_cols]
        headers = [
            next(h for c, h in table_columns if c == col) for col in ordered_cols
        ]

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in combined.iterrows():
            values = []
            for col in ordered_cols:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.6g}")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        logging.error(f"Failed to write markdown data table for {save_path}: {e}")


bpw_lt = {
    "Llama31_8B IQ1_M": 2.15,
    "Llama31_8B IQ1_S": 2.00,
    "Llama31_8B IQ2_M": 2.93,
    "Llama31_8B IQ2_S": 2.74,
    "Llama31_8B IQ2_XS": 2.59,
    "Llama31_8B IQ2_XXS": 2.38,
    "Llama31_8B IQ3_M": 3.76,
    "Llama31_8B IQ3_S": 3.66,
    "Llama31_8B IQ3_XS": 3.50,
    "Llama31_8B IQ3_XXS": 3.25,
    "Llama31_8B IQ4_NL": 4.65,
    "Llama31_8B IQ4_XS": 4.42,
    "Llama31_8B Q2_K_S": 2.97,
    "Llama31_8B Q2_K": 3.16,
    "Llama31_8B Q3_K_L": 4.30,
    "Llama31_8B Q3_K_M": 4.00,
    "Llama31_8B Q3_K_S": 3.64,
    "Llama31_8B Q4_0": 4.65,
    "Llama31_8B Q4_1": 5.10,
    "Llama31_8B Q4_K_M": 4.89,
    "Llama31_8B Q4_K_S": 4.67,
    "Llama31_8B Q5_0": 5.59,
    "Llama31_8B Q5_1": 6.04,
    "Llama31_8B Q5_K_M": 5.70,
    "Llama31_8B Q5_K_S": 5.57,
    "Llama31_8B Q6_K": 6.56,
    "Llama31_8B Q8_0": 8.50,
    "Llama31_8B F16": 16.00,
    "Llama32_3B IQ1_M": 2.28,
    "Llama32_3B IQ1_S": 2.14,
    "Llama32_3B IQ2_M": 3.04,
    "Llama32_3B IQ2_S": 2.85,
    "Llama32_3B IQ2_XS": 2.72,
    "Llama32_3B IQ2_XXS": 2.51,
    "Llama32_3B IQ3_M": 3.96,
    "Llama32_3B IQ3_S": 3.82,
    "Llama32_3B IQ3_XS": 3.66,
    "Llama32_3B IQ3_XXS": 3.34,
    "Llama32_3B IQ4_NL": 4.75,
    "Llama32_3B IQ4_XS": 4.54,
    "Llama32_3B Q2_K_S": 3.15,
    "Llama32_3B Q2_K": 3.38,
    "Llama32_3B Q3_K_L": 4.50,
    "Llama32_3B Q3_K_M": 4.18,
    "Llama32_3B Q3_K_S": 3.82,
    "Llama32_3B Q4_0": 4.77,
    "Llama32_3B Q4_1": 5.19,
    "Llama32_3B Q4_K_M": 5.01,
    "Llama32_3B Q4_K_S": 4.78,
    "Llama32_3B Q5_0": 5.64,
    "Llama32_3B Q5_1": 6.07,
    "Llama32_3B Q5_K_M": 5.76,
    "Llama32_3B Q5_K_S": 5.63,
    "Llama32_3B Q6_K": 6.56,
    "Llama32_3B Q8_0": 8.50,
    "Llama32_3B F16": 16.00,
}

mlc_sort = {
    "q3f16_0": 1,
    "q3f16_1": 2,
    "q4f16_0": 3,
    "q4f16_1": 4,
    "q4f32_1": 5,
    "q0f16": 6,
    "q0f32": 7,
}

executorch_sort = {
    "SpinQuant": 1,
    "BF16": 2,
}


def load_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def get_bpw(model: str, quant: str) -> int:
    key = f"{model} {quant}"
    return bpw_lt[key]


MMLU = "MMLU"
IFEVAL = "IFEval"

defined_models = ["Llama31_8B", "Llama32_3B"]
defined_benchmarks = [MMLU, IFEVAL]

FAMILY_LABEL_TO_MODEL_ID = {"Llama 3.1 8B": "Llama31_8B", "Llama 3.2 3B": "Llama32_3B"}

ENGINE_SYMBOLS = {"llama.cpp": "●", "ExecuTorch": "▲", "MLC": "■"}


def quant_engine_group_and_order(quant: str, family: Optional[str] = None):
    """Return a tuple (group, order_value) for sorting quantizations"""

    # Try llama.cpp via bpw_lt
    def _find_bpw_for_quant(q: str) -> Optional[float]:
        if family:
            model_id = FAMILY_LABEL_TO_MODEL_ID.get(family)
            if model_id is not None:
                key = f"{model_id} {q}"
                if key in bpw_lt:
                    return bpw_lt[key]
        for model_id in ["Llama31_8B", "Llama32_3B"]:
            key = f"{model_id} {q}"
            if key in bpw_lt:
                return bpw_lt[key]
        return None

    bpw = _find_bpw_for_quant(quant)
    if bpw is not None:
        return (0, float(bpw))

    # ExecuTorch
    if quant in executorch_sort:
        return (1, int(executorch_sort[quant]))

    # MLC
    if quant in mlc_sort:
        return (2, int(mlc_sort[quant]))

    return (3, str(quant))


def prepend_engine_symbol_to_quant(quant: str, family: Optional[str] = None) -> str:
    group_idx, _ = quant_engine_group_and_order(quant, family)
    if group_idx == 0:
        symbol = ENGINE_SYMBOLS.get("llama.cpp", "")
    elif group_idx == 1:
        symbol = ENGINE_SYMBOLS.get("ExecuTorch", "")
    elif group_idx == 2:
        symbol = ENGINE_SYMBOLS.get("MLC", "")
    else:
        symbol = ""
    return f"{symbol} {quant}" if symbol else str(quant)


def load_eval_data(folder: str):
    json_files = list(Path("./data/" + folder).glob("*.json"))
    datas = []
    for file in json_files:
        model, benchmark, quant = file.name.split("-")
        quant = quant.split(".", maxsplit=1)[0]
        if model in defined_models and benchmark in defined_benchmarks:
            data = load_json(file)
            datas.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "quant": quant,
                    "data": data,
                    "bpw": get_bpw(model, quant),
                    "mmlu_acc": data.get("mmlu/acc", None),
                    "mmlu_stderr": data.get("mmlu/acc_stderr", None),
                    "ifeval_inst_loose_acc": data.get(
                        "ifeval/inst_level_loose_acc", None
                    ),
                    "ifeval_inst_strict_acc": data.get(
                        "ifeval/inst_level_strict_acc", None
                    ),
                    "ifeval_prompt_loose_acc": data.get(
                        "ifeval/prompt_level_loose_acc", None
                    ),
                    "ifeval_prompt_loose_stderr": data.get(
                        "ifeval/prompt_level_loose_acc_stderr", None
                    ),
                    "ifeval_prompt_strict_acc": data.get(
                        "ifeval/prompt_level_strict_acc", None
                    ),
                    "ifeval_prompt_strict_stderr": data.get(
                        "ifeval/prompt_level_strict_acc_stderr", None
                    ),
                }
            )
    return datas


def make_plot(
    data_list: list[any],
    x_key: str,
    x_title: str,
    y_key: str,
    y_title: str,
    save_path: Optional[str] = None,
):
    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        fig, ax = plt.subplots(figsize=(12, 7))

        draggable_texts = []

        all_accs = np.concatenate([data["df"][y_key].values for data in data_list])
        y_range = all_accs.max() - all_accs.min()
        threshold = y_range * 0.1

        for i, data in enumerate(data_list):
            df = data["df"]
            color = data.get("color", f"C{i}")
            name = data.get("name", f"Dataset {i + 1}")

            pareto_quants = set()
            try:
                if {"quant", x_key, y_key}.issubset(df.columns):
                    df_ann = df[[x_key, y_key, "quant"]].dropna().copy()
                    if not df_ann.empty:
                        idxs = df_ann.groupby(x_key)[y_key].idxmax()
                        df_unique = df_ann.loc[idxs].sort_values(
                            by=x_key, ascending=True
                        )
                        current_max_y = -np.inf
                        for _, r in df_unique.iterrows():
                            y_val = float(r[y_key])
                            if y_val >= current_max_y - 1e-12:
                                pareto_quants.add(str(r["quant"]))
                                current_max_y = y_val
            except Exception as e:
                logging.debug(f"Pareto membership computation failed: {e}")

            ax.scatter(
                df[x_key], df[y_key], s=80, alpha=0.7, color=color, label=name, zorder=2
            )

            for _, row in df.iterrows():
                text = ax.text(
                    x=row[x_key] + 0.025,
                    y=row[y_key] + 0.015,
                    s=row["quant"],
                    fontsize=11,
                    fontweight=(
                        "bold" if str(row["quant"]) in pareto_quants else "normal"
                    ),
                    rotation=45,
                    rotation_mode="anchor",
                    ha="right",
                    va="bottom",
                    color=color,
                    zorder=100,
                )
                dt = DraggableText(text, row[x_key], row[y_key], threshold, ax, color)
                draggable_texts.append(dt)
                dt.update_line()

            if "baseline" in data and data["baseline"] is not None:
                baseline_label, baseline_acc = data["baseline"]
                plt.axhline(
                    y=baseline_acc,
                    color=color,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )

                x_min, x_max = plt.xlim()
                label_x = x_max - (x_max - x_min) * 0.005
                label_y = baseline_acc
                plt.text(
                    label_x,
                    label_y,
                    baseline_label,
                    color=color,
                    ha="right",
                    va="bottom",
                )

            try:
                df_pf = df[[x_key, y_key]].dropna().copy()
                if not df_pf.empty:
                    df_pf = df_pf.groupby(x_key, as_index=False)[y_key].max()
                    df_pf = df_pf.sort_values(by=x_key, ascending=True)

                    xs = []
                    ys = []
                    current_max_y = -np.inf
                    for _, r in df_pf.iterrows():
                        x_val = float(r[x_key])
                        y_val = float(r[y_key])
                        if y_val >= current_max_y - 1e-12:
                            xs.append(x_val)
                            ys.append(y_val)
                            current_max_y = y_val

                    if len(xs) >= 2:
                        ax.plot(
                            xs,
                            ys,
                            linestyle="--",
                            linewidth=1.0,
                            color=color,
                            alpha=0.9,
                            zorder=1,
                        )
            except Exception as e:
                logging.debug(f"Pareto front computation failed: {e}")

        ax.set_xscale("log", base=8)
        ax.set_xlabel(x_title, fontsize=16)
        ax.set_ylabel(y_title, fontsize=16)
        ax.grid(True, alpha=0.3, linestyle="--", axis="both", which="both", zorder=0)

        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(current_ymin - 0.02, current_ymax + 0.04)

        ax.set_xlim([1.8, 9.5])

        min_bpw = min([data["df"][x_key].min() for data in data_list])
        max_bpw = max([data["df"][x_key].max() for data in data_list])

        tick_locs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tick_locs = [x for x in tick_locs if min_bpw <= x <= max_bpw * 1.5]
        ax.set_xticks(tick_locs)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

        if len(data_list) > 1:
            ax.legend(loc="best", fontsize=12)

        fig.tight_layout(pad=0.5)

    plt.show()
    if save_path:
        try:
            logging.debug(f"Saving plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            _save_markdown_table_for_plot(
                save_path,
                data_list,
                [
                    (x_key, x_title if isinstance(x_title, str) else x_key),
                    (y_key, y_title if isinstance(y_title, str) else y_key),
                    ("quant", "quant"),
                ],
            )
            logging.debug("Plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving plot: {e}")

    return fig, ax


class DataEntry(TypedDict):
    data: list[any]
    color: str
    name: str


def make_mmlu_plot(entry_data: list[DataEntry], save_path: str):
    datas = []
    for entry in entry_data:
        plot_data = [
            {"bpw": _["bpw"], "acc": _["mmlu_acc"], "quant": _["quant"]}
            for _ in entry["data"]
            if _["bpw"] != 16.00
        ]
        df = pd.DataFrame(plot_data)
        datas.append(
            {
                "df": df,
                # 'baseline': ('f16', 0.6),
                "baseline": None,
                "color": entry["color"],
                "name": entry["name"],
            }
        )
    make_plot(datas, "bpw", "Bity na wagę", "acc", "Dokładność", save_path=save_path)


def make_ifeval_plot(entry_data: list[DataEntry], save_path: str):
    datas = []
    for entry in entry_data:
        acc_metric = "ifeval_inst_loose_acc"
        plot_data = [
            {"bpw": _["bpw"], "acc": _[acc_metric], "quant": _["quant"]}
            for _ in entry["data"]
            if _["bpw"] != 16.00
        ]
        df = pd.DataFrame(plot_data)
        datas.append(
            {
                "df": df,
                # 'baseline': ('f16', 0.6),
                "baseline": None,
                "color": entry["color"],
                "name": entry["name"],
            }
        )
    make_plot(datas, "bpw", "Bity na wagę", "acc", "Dokładność", save_path)


def make_mmlu_vs_ifeval_plot(
    mmlu_entry_data: list[DataEntry], ifeval_entry_data: list[DataEntry], save_path: str
):
    """Create a scatter plot with IFEval accuracy on the X axis and MMLU accuracy on the Y axis."""
    ifeval_by_name = {e["name"]: e for e in ifeval_entry_data}

    datas = []
    for mmlu_entry in mmlu_entry_data:
        name = mmlu_entry["name"]
        color = mmlu_entry["color"]
        if name not in ifeval_by_name:
            continue

        ifeval_entry = ifeval_by_name[name]

        mmlu_map = {
            d["quant"]: d.get("mmlu_acc")
            for d in mmlu_entry["data"]
            if d.get("bpw") != 16.00 and d.get("mmlu_acc") is not None
        }
        acc_metric = "ifeval_inst_loose_acc"
        ifeval_map = {
            d["quant"]: d.get(acc_metric)
            for d in ifeval_entry["data"]
            if d.get("bpw") != 16.00 and d.get(acc_metric) is not None
        }

        common_quants = sorted(set(mmlu_map.keys()) & set(ifeval_map.keys()))
        if not common_quants:
            continue

        plot_rows = [
            {"x": float(ifeval_map[q]), "y": float(mmlu_map[q]), "quant": q}
            for q in common_quants
        ]

        df = pd.DataFrame(plot_rows)
        datas.append({"df": df, "color": color, "name": name})

    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        fig, ax = plt.subplots(figsize=(12, 7))

        draggable_texts = []

        if datas:
            all_accs = np.concatenate([d["df"]["y"].values for d in datas])
            y_range = float(all_accs.max() - all_accs.min()) if len(all_accs) else 0.0
        else:
            y_range = 0.0
        threshold = y_range * 0.1 if y_range > 0 else 0.05

        for i, data in enumerate(datas):
            df = data["df"]
            color = data.get("color", f"C{i}")
            name = data.get("name", f"Dataset {i + 1}")

            ax.scatter(
                df["x"], df["y"], s=80, alpha=0.7, color=color, label=name, zorder=2
            )

            pareto_quants = set()
            try:
                if {"quant", "x", "y"}.issubset(df.columns):
                    df_ann = df[["x", "y", "quant"]].dropna().copy()
                    if not df_ann.empty:
                        idxs = df_ann.groupby("x")["y"].idxmax()
                        df_unique = df_ann.loc[idxs].sort_values(by="x", ascending=True)
                        current_max_y = -np.inf
                        for _, r in df_unique.iterrows():
                            if float(r["y"]) >= current_max_y - 1e-12:
                                pareto_quants.add(str(r["quant"]))
                                current_max_y = float(r["y"])
            except Exception as e:
                logging.debug(f"Pareto membership (MMLU vs IFEval) failed: {e}")

            for _, row in df.iterrows():
                text = ax.text(
                    x=row["x"] + 0.01,
                    y=row["y"] + 0.01,
                    s=row["quant"],
                    fontsize=11,
                    fontweight=(
                        "bold" if str(row["quant"]) in pareto_quants else "normal"
                    ),
                    rotation=45,
                    rotation_mode="anchor",
                    ha="right",
                    va="bottom",
                    color=color,
                    zorder=100,
                )
                dt = DraggableText(text, row["x"], row["y"], threshold, ax, color)
                draggable_texts.append(dt)
                dt.update_line()

            try:
                df_pf = df[["x", "y"]].dropna().copy()
                if not df_pf.empty:
                    df_pf = df_pf.groupby("x", as_index=False)["y"].max()
                    df_pf = df_pf.sort_values(by="x", ascending=True)

                    xs = []
                    ys = []
                    current_max_y = -np.inf
                    for _, r in df_pf.iterrows():
                        x_val = float(r["x"])
                        y_val = float(r["y"])
                        if y_val >= current_max_y - 1e-12:
                            xs.append(x_val)
                            ys.append(y_val)
                            current_max_y = y_val

                    if len(xs) >= 2:
                        ax.plot(
                            xs,
                            ys,
                            linestyle="--",
                            linewidth=1.0,
                            color=color,
                            alpha=0.9,
                            zorder=1,
                        )
            except Exception as e:
                logging.debug(f"Pareto front (MMLU vs IFEval) computation failed: {e}")

        ax.set_xlabel("Dokładność (IFEval)", fontsize=16)
        ax.set_ylabel("Dokładność (MMLU)", fontsize=16)
        ax.grid(True, alpha=0.3, linestyle="--", axis="both", which="both", zorder=0)

        if datas:
            all_x = np.concatenate([d["df"]["x"].values for d in datas])
            all_y = np.concatenate([d["df"]["y"].values for d in datas])
            xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
            ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            xpad = (xmax - xmin) * 0.05 if xmax > xmin else 0.02
            ypad = (ymax - ymin) * 0.05 if ymax > ymin else 0.02
            ax.set_xlim(max(0.0, xmin - xpad), min(1.0, xmax + xpad))
            ax.set_ylim(max(0.0, ymin - ypad), min(1.0, ymax + ypad + 0.02))
        else:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        if len(datas) > 1:
            ax.legend(loc="best", fontsize=12)

        fig.tight_layout(pad=0.5)

    plt.show()
    if save_path:
        try:
            logging.debug(f"Saving plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            _save_markdown_table_for_plot(
                save_path, datas, [("x", "x"), ("y", "y"), ("quant", "quant")]
            )
            logging.debug("Plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving plot: {e}")

    return fig, ax


def load_excel_worksheet(
    excel_file_path: str, worksheet_name: str
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(excel_file_path, sheet_name=worksheet_name)
        return df
    except FileNotFoundError:
        logging.debug(f"Error: File '{excel_file_path}' not found")
        return None
    except ValueError:
        logging.debug(
            f"Error: Worksheet '{worksheet_name}' not found in '{excel_file_path}'"
        )
        return None
    except Exception as e:
        logging.debug(f"An error occurred: {str(e)}")
        return None


def sort_by_quant(x):
    quant = x.split(" ")[3]
    if "I" in quant:
        quant = quant.replace("I", "") + "I"
    return quant


def make_tps_plot(
    df_data: pd.DataFrame,
    x_label: str,
    y_label: str,
    save_path: Optional[str] = None,
    device_model: Optional[str] = None,
    family: Optional[str] = None,
):
    filtered_data = df_data.copy()
    if device_model:
        filtered_data = filtered_data[filtered_data["deviceModel"] == device_model]
    if family:
        filtered_data = filtered_data[filtered_data["family"] == family]

    if filtered_data.empty:
        logging.debug(f"No data found for device_model={device_model}, family={family}")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 20))

    application_order = ["llama.cpp", "ExecuTorch", "MLC"]
    application_colors = {
        "llama.cpp": "#1f77b4",  # blue
        "ExecuTorch": "#ff7f0e",  # orange
        "MLC": "#2ca02c",  # green
    }

    grouped_data = (
        filtered_data.groupby(["application", "model"])["tps"].apply(list).reset_index()
    )
    grouped_data["quant"] = (
        grouped_data["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )
    grouped_data["application_order"] = grouped_data["application"].map(
        {app: i for i, app in enumerate(application_order)}
    )
    grouped_data = grouped_data.sort_values(
        ["application_order", "application"], kind="mergesort"
    )

    def _sort_within_app(df_app: pd.DataFrame) -> pd.DataFrame:
        def _sort_key_with_family(q):
            if family:
                return quant_engine_group_and_order(q, family)
            else:
                return quant_engine_group_and_order(q)

        return df_app.sort_values(
            by="quant", key=lambda s: s.map(_sort_key_with_family)
        )

    grouped_data = grouped_data.groupby("application", group_keys=False).apply(
        _sort_within_app
    )

    positions = []
    box_data = []
    labels = []
    colors = []
    current_pos = 1

    for app in application_order:
        app_data = grouped_data[grouped_data["application"] == app]
        if not app_data.empty:
            for _, row in app_data.iterrows():
                box_data.append(row["tps"])
                quant_token = row["model"].split(" ")[-1]
                symbol = ENGINE_SYMBOLS.get(app, "")
                label_text = f"{symbol} {quant_token}" if symbol else quant_token
                labels.append(label_text)
                colors.append(application_colors[app])
                positions.append(current_pos)
                current_pos += 1
            current_pos += 0.5

    bp = ax.boxplot(
        box_data, positions=positions, patch_artist=True, widths=0.8, vert=False
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    whisker_cap_colors = [color for color in colors for _ in range(2)]

    for whisker, color in zip(bp["whiskers"], whisker_cap_colors):
        whisker.set_color(color)

    for cap, color in zip(bp["caps"], whisker_cap_colors):
        cap.set_color(color)

    for median, color in zip(bp["medians"], colors):
        median.set_color(color)

    ax.set_xlabel(y_label, fontsize=16)
    ax.set_ylabel(x_label, fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.set_yticks([pos for pos, label in zip(positions, labels) if label])
    ax.set_yticklabels([label for label in labels if label])

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=application_colors[app], alpha=0.7)
        for app in application_order
        if app in df_data["application"].values
    ]
    legend_labels = [
        app for app in application_order if app in df_data["application"].values
    ]
    ax.legend(
        legend_elements,
        legend_labels,
        title="Silnik inferencji",
        loc="upper left",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    if save_path:
        try:
            logging.debug(f"Saving plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            try:
                flat_rows = []
                for _, row in grouped_data.iterrows():
                    app = row["application"]
                    model = row["model"]
                    quant_token = row["quant"]
                    tps_list = row["tps"] if isinstance(row["tps"], list) else []
                    for t in tps_list:
                        flat_rows.append(
                            {
                                "application": app,
                                "model": model,
                                "quant": quant_token,
                                "tps": float(t) if t is not None else None,
                            }
                        )
                if flat_rows:
                    df_export = pd.DataFrame(flat_rows)
                    _save_markdown_table_for_plot(
                        save_path,
                        [{"df": df_export}],
                        [
                            ("application", "application"),
                            ("model", "model"),
                            ("quant", "quant"),
                            ("tps", "tps"),
                        ],
                    )
            except Exception as ex:
                logging.error(f"Failed to export TPS markdown table: {ex}")
            logging.debug("Plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving plot: {e}")

    return fig, ax


def make_model_based_tps_plots(df_data: pd.DataFrame, save_path_dir: str):
    """Create TPS plots for each model family"""
    df_data = df_data[df_data["inputTokens"] == 64]
    families = df_data["family"].unique()

    logging.debug(
        f"Creating plots for {len(families)} model families with all smartphones..."
    )

    for family in families:
        family_clean = family.replace(" ", "_").replace(".", "")
        filename = f"tps_plot_{family_clean}_all_devices.svg"
        save_path = os.path.join(save_path_dir, filename)

        logging.debug(f"Creating plot: {filename}")
        fig, _ = make_tps_plot_per_family(
            df_data=df_data,
            x_label="Kwantyzacja",
            y_label="Prędkość generowania (tokeny/s)",
            save_path=save_path,
            family=family,
        )

        if fig is not None:
            plt.close(fig)


def make_tps_plot_per_family(
    df_data: pd.DataFrame,
    x_label: str,
    y_label: str,
    save_path: Optional[str] = None,
    family: Optional[str] = None,
):
    """Create TPS plot for a specific model family with data from all smartphones.

    Modified: Draw side-by-side subplots per smartphone with a shared Y-axis.
    """
    filtered_data = df_data.copy()
    if family:
        filtered_data = filtered_data[filtered_data["family"] == family]

    if filtered_data.empty:
        logging.debug(f"No data found for family={family}")
        return None, None

    application_order = ["llama.cpp", "ExecuTorch", "MLC"]

    application_colors = {
        "llama.cpp": "#1f77b4",  # blue
        "ExecuTorch": "#ff7f0e",  # orange
        "MLC": "#2ca02c",  # green
    }

    device_names = {
        "CPH2581": "OnePlus 12",
        "IN2013": "OnePlus 8",
        "SM-A256B": "Samsung A25",
    }

    grouped_data = (
        filtered_data.groupby(["application", "model", "deviceModel"])["tps"]
        .apply(list)
        .reset_index()
    )

    if grouped_data.empty:
        logging.debug("No grouped TPS data available for plotting.")
        return None, None

    grouped_data["quant"] = (
        grouped_data["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )

    grouped_data["application_order"] = grouped_data["application"].map(
        {app: i for i, app in enumerate(application_order)}
    )
    grouped_data = grouped_data.sort_values(
        ["application_order", "application"], kind="mergesort"
    )

    def _sort_within_app(df_app: pd.DataFrame) -> pd.DataFrame:
        return df_app.sort_values(
            by="quant",
            key=lambda s: s.map(lambda q: quant_engine_group_and_order(q, family)),
        )

    grouped_data = grouped_data.groupby("application", group_keys=False).apply(
        _sort_within_app
    )

    ordered_unique = grouped_data.drop_duplicates(subset=["application", "model"]).copy()

    positions_all = []
    labels_all = []
    id_tuples = []
    current_pos = 1
    for app in application_order:
        app_rows = ordered_unique[ordered_unique["application"] == app]
        if app_rows.empty:
            continue
        for _, row in app_rows.iterrows():
            model_name = row["model"]
            quant_token = str(model_name).split(" ")[-1]
            symbol = ENGINE_SYMBOLS.get(app, "")
            label_text = f"{symbol} {quant_token}" if symbol else quant_token
            positions_all.append(current_pos)
            labels_all.append(label_text)
            id_tuples.append((app, model_name))
            current_pos += 1
        current_pos += 1.0

    devices_present = [
        d for d in ["CPH2581", "IN2013", "SM-A256B"] if d in filtered_data["deviceModel"].unique()
    ]
    if not devices_present:
        devices_present = list(filtered_data["deviceModel"].unique())

    n_devs = len(devices_present)
    fig_width = max(12, 6 * n_devs)
    fig, axes = plt.subplots(1, n_devs, sharey=True, figsize=(fig_width, 10))
    if n_devs == 1:
        axes = [axes]

    for ax, device in zip(axes, devices_present):
        box_data = []
        positions = []
        colors = []

        for pos, (app, model_name) in zip(positions_all, id_tuples):
            sel = grouped_data[
                (grouped_data["application"] == app)
                & (grouped_data["model"] == model_name)
                & (grouped_data["deviceModel"] == device)
            ]
            if sel.empty:
                continue
            box_data.append(sel.iloc[0]["tps"])
            positions.append(pos)
            colors.append(application_colors.get(app, "gray"))

        if box_data:
            bp = ax.boxplot(
                box_data, positions=positions, patch_artist=True, widths=0.8, vert=False
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)
                patch.set_linewidth(1.3)
            whisker_cap_colors = [color for color in colors for _ in range(2)]
            for whisker, color in zip(bp["whiskers"], whisker_cap_colors):
                whisker.set_color(color)
                whisker.set_linewidth(1.2)
            for cap, color in zip(bp["caps"], whisker_cap_colors):
                cap.set_color(color)
                cap.set_linewidth(1.2)
            for median, color in zip(bp["medians"], colors):
                median.set_color(color)
                median.set_linewidth(1.5)

            try:
                x_left, x_right = ax.get_xlim()
                axis_span = float(x_right - x_left)
                min_span = axis_span * 0.02 if axis_span > 0 else 0.1
                for i, patch in enumerate(bp["boxes"]):
                    verts = patch.get_path().vertices
                    xs = verts[:, 0]
                    ys = verts[:, 1]
                    x_span = float(np.nanmax(xs) - np.nanmin(xs)) if len(xs) else 0.0
                    if x_span < min_span and i < len(positions) and i < len(colors):
                        color = colors[i]
                        med_line = bp["medians"][i]
                        x_med_vals = med_line.get_xdata()
                        x_center = float(np.nanmean(x_med_vals)) if len(x_med_vals) else float(np.nanmean(xs))
                        if len(ys):
                            y_min = float(np.nanmin(ys))
                            y_max = float(np.nanmax(ys))
                        else:
                            y_center = positions[i]
                            y_min = y_center - 0.4
                            y_max = y_center + 0.4
                        rect_height = y_max - y_min
                        rect = Rectangle(
                            (x_center - min_span / 2.0, y_min),
                            min_span,
                            rect_height,
                            facecolor=color,
                            edgecolor="black",
                            alpha=0.6,
                            zorder=patch.get_zorder() - 0.1,
                        )
                        ax.add_patch(rect)
            except Exception:
                pass

        ax.set_title(device_names.get(device, device), fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel(y_label, fontsize=16)

    axes[0].set_ylabel(x_label, fontsize=16)
    axes[0].set_yticks(positions_all)
    axes[0].set_yticklabels(labels_all, ha="right")

    if positions_all:
        min_pos = min(positions_all) - 1
        max_pos = max(positions_all) + 1
        for ax in axes:
            ax.set_ylim(min_pos, max_pos)

    legend_elements = [
        Patch(facecolor=application_colors.get(app, "gray"), edgecolor="black", alpha=0.8)
        for app in application_order
        if app in grouped_data["application"].unique()
    ]
    legend_labels = [app for app in application_order if app in grouped_data["application"].unique()]
    if legend_elements:
        axes[-1].legend(
            legend_elements,
            legend_labels,
            title="Silnik",
            loc="upper right",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

    if save_path:
        try:
            logging.debug(f"Saving plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            try:
                flat_rows = []
                for _, row in grouped_data.iterrows():
                    app = row["application"]
                    model = row["model"]
                    quant_token = row["quant"]
                    device = row["deviceModel"]
                    tps_list = row["tps"] if isinstance(row["tps"], list) else []
                    for t in tps_list:
                        flat_rows.append(
                            {
                                "application": app,
                                "deviceModel": device,
                                "model": model,
                                "quant": quant_token,
                                "tps": float(t) if t is not None else None,
                            }
                        )
                if flat_rows:
                    df_export = pd.DataFrame(flat_rows)
                    _save_markdown_table_for_plot(
                        save_path,
                        [{"df": df_export}],
                        [
                            ("application", "application"),
                            ("deviceModel", "deviceModel"),
                            ("model", "model"),
                            ("quant", "quant"),
                            ("tps", "tps"),
                        ],
                    )
            except Exception as ex:
                logging.error(f"Failed to export per-family TPS markdown table: {ex}")
            logging.debug("Plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving plot: {e}")

    return fig, axes


def make_prefill_vs_tps_scatter_plot(
    df_data: pd.DataFrame, save_path: str, model_filter: str = None
):
    """Create single scatter plot with prefillSpeed vs tps, colored by application and shaped by deviceModel"""
    if model_filter:
        df_data = df_data[df_data["family"] == model_filter].copy()
        if df_data.empty:
            logging.debug(f"No data found for model family: {model_filter}")
            return

    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        fig, ax = plt.subplots(figsize=(12, 8))

        application_colors = {
            "llama.cpp": "#1f77b4",  # blue
            "ExecuTorch": "#ff7f0e",  # orange
            "MLC": "#2ca02c",  # green
        }

        unique_devices = df_data["deviceModel"].unique()
        device_markers = [
            "o",
            "*",
            "^",
            "v",
            "<",
            ">",
            "D",
            "p",
            "s",
            "h",
            "H",
            "+",
            "x",
        ]
        device_symbols = {
            device: device_markers[i % len(device_markers)]
            for i, device in enumerate(unique_devices)
        }

        device_name_map = {
            "CPH2581": "OnePlus 12",
            "IN2013": "OnePlus 8",
            "SM-A256B": "Samsung A25",
        }

        marker_to_text = {
            "o": "●",
            "*": "★",
            "^": "▲",
            "v": "▼",
            "<": "◀",
            ">": "▶",
            "D": "◆",
            "p": "⬟",
            "s": "■",
            "h": "⬢",
            "H": "⬡",
            "+": "✚",
            "x": "✕",
        }

        aggregated_data = []

        for device in unique_devices:
            device_data = df_data[df_data["deviceModel"] == device]

            for app in device_data["application"].unique():
                app_data = device_data[device_data["application"] == app]

                if not app_data.empty:
                    grouped = (
                        app_data.groupby(["family", "model"])
                        .agg({"prefillSpeed": "mean", "tps": "mean"})
                        .reset_index()
                    )

                    for _, row in grouped.iterrows():
                        model_parts = row["model"].split(" ")
                        if len(model_parts) >= 3:
                            annotation = model_parts[-1]
                        else:
                            annotation = row["model"]

                        aggregated_data.append(
                            {
                                "x": row["prefillSpeed"],
                                "y": row["tps"],
                                "annotation": annotation,
                                "family": row["family"],
                                "model": row["model"],
                                "application": app,
                                "device": device,
                                "color": application_colors.get(app, "gray"),
                                "marker": device_symbols[device],
                            }
                        )

        for app in df_data["application"].unique():
            for device in unique_devices:
                app_device_points = [
                    p
                    for p in aggregated_data
                    if p["application"] == app and p["device"] == device
                ]

                if app_device_points:
                    x_values = [p["x"] for p in app_device_points]
                    y_values = [p["y"] for p in app_device_points]

                    ax.scatter(
                        x_values,
                        y_values,
                        c=application_colors.get(app, "gray"),
                        marker=device_symbols[device],
                        s=80,
                        alpha=0.7,
                        edgecolors="black",
                        linewidths=0.5,
                    )

        draggable_texts = []
        if aggregated_data:
            all_y_vals = np.array([p["y"] for p in aggregated_data], dtype=float)
            y_range = (
                float(np.nanmax(all_y_vals) - np.nanmin(all_y_vals))
                if len(all_y_vals)
                else 0.0
            )
        else:
            y_range = 0.0
        drag_threshold = y_range * 0.1 if y_range > 0 else 0.05

        if aggregated_data:
            aggregated_data.sort(key=lambda p: p["y"], reverse=True)

            annotation_positions = []

            for point in aggregated_data:
                x, y = point["x"], point["y"]
                device = point["device"]
                device_marker = device_symbols[device]
                device_symbol_text = marker_to_text.get(device_marker, device_marker)
                annotation_text = f"{device_symbol_text} {point['annotation']}"

                potential_positions = [
                    (x, y, 5, 5),  # Top-right of point
                    (x, y, -5, 5),  # Top-left of point
                    (x, y, 5, -5),  # Bottom-right of point
                    (x, y, -5, -5),  # Bottom-left of point
                    (x, y, 10, 10),  # Further top-right
                    (x, y, -10, 10),  # Further top-left
                    (x, y, 10, -10),  # Further bottom-right
                    (x, y, -10, -10),  # Further bottom-left
                ]

                placed = False
                for px, py, offset_x, offset_y in potential_positions:
                    too_close = False
                    for ann_x, ann_y in annotation_positions:
                        display_x = ax.transData.transform([(px, py)])[0][0]
                        display_y = ax.transData.transform([(px, py)])[0][1]
                        display_ann_x = ax.transData.transform([(ann_x, ann_y)])[0][0]
                        display_ann_y = ax.transData.transform([(ann_x, ann_y)])[0][1]

                        distance = (
                            (display_x - display_ann_x) ** 2
                            + (display_y - display_ann_y) ** 2
                        ) ** 0.5

                        if distance < 30:
                            too_close = True
                            break

                    if not too_close:
                        ha = "left" if offset_x >= 0 else "right"
                        va = "bottom" if offset_y >= 0 else "top"

                        ann = ax.annotate(
                            annotation_text,
                            xy=(px, py),
                            xytext=(offset_x, offset_y),
                            textcoords="offset points",
                            fontsize=9,
                            color=point["color"],
                            alpha=0.9,
                            ha=ha,
                            va=va,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.8,
                                edgecolor="gray",
                                linewidth=0.5,
                            ),
                        )
                        dt = DraggableText(
                            ann,
                            px,
                            py,
                            drag_threshold,
                            ax,
                            point["color"],
                            use_connector=False,
                        )
                        draggable_texts.append(dt)
                        dt.update_line()
                        annotation_positions.append((px, py))
                        placed = True
                        break

                if not placed and len(annotation_text) > 3:
                    short_text = annotation_text
                    for px, py, offset_x, offset_y in potential_positions:
                        too_close = False
                        for ann_x, ann_y in annotation_positions:
                            display_x = ax.transData.transform([(px, py)])[0][0]
                            display_y = ax.transData.transform([(px, py)])[0][1]
                            display_ann_x = ax.transData.transform([(ann_x, ann_y)])[0][
                                0
                            ]
                            display_ann_y = ax.transData.transform([(ann_x, ann_y)])[0][
                                1
                            ]

                            distance = (
                                (display_x - display_ann_x) ** 2
                                + (display_y - display_ann_y) ** 2
                            ) ** 0.5

                            if distance < 25:
                                too_close = True
                                break

                        if not too_close:
                            ha = "left" if offset_x >= 0 else "right"
                            va = "bottom" if offset_y >= 0 else "top"

                            ann = ax.annotate(
                                short_text,
                                xy=(px, py),
                                xytext=(offset_x, offset_y),
                                textcoords="offset points",
                                fontsize=8,
                                color=point["color"],
                                alpha=0.9,
                                ha=ha,
                                va=va,
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    alpha=0.8,
                                    edgecolor="gray",
                                    linewidth=0.5,
                                ),
                            )
                            dt = DraggableText(
                                ann,
                                px,
                                py,
                                drag_threshold,
                                ax,
                                point["color"],
                                use_connector=False,
                            )
                            draggable_texts.append(dt)
                            dt.update_line()
                            annotation_positions.append((px, py))
                            break

        app_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=application_colors.get(app, "gray"),
                markersize=10,
                label=app,
                markeredgecolor="black",
            )
            for app in df_data["application"].unique()
        ]

        device_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=device_symbols[device],
                color="w",
                markerfacecolor="gray",
                markersize=10,
                label=device_name_map.get(device, device),
                markeredgecolor="black",
            )
            for device in unique_devices
        ]

        app_legend = ax.legend(
            handles=app_legend_elements,
            title="Silnik inferencji",
            loc="lower right",
            fontsize=10,
            bbox_to_anchor=(1, 0),
        )
        device_legend = ax.legend(
            handles=device_legend_elements,
            title="Smartfon",
            loc="lower right",
            fontsize=10,
            bbox_to_anchor=(1, 0.13),
        )

        ax.add_artist(app_legend)
        ax.add_artist(device_legend)

        ax.set_xlabel("Prędkość wstępnego przetwarzania (tokeny/s)", fontsize=16)
        ax.set_ylabel("Prędkość generowania (tokeny/s)", fontsize=16)

        ax.grid(True, linestyle="--", alpha=0.7)

        ax.ticklabel_format(style="plain", axis="x")

        plt.tight_layout()
        plt.show()

        try:
            logging.debug(f"Saving scatter plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            try:
                if aggregated_data:
                    df_export = pd.DataFrame(aggregated_data)
                    _save_markdown_table_for_plot(
                        save_path,
                        [{"df": df_export}],
                        [
                            ("application", "application"),
                            ("device", "deviceModel"),
                            ("family", "family"),
                            ("model", "model"),
                            ("annotation", "quant"),
                            ("x", "prefillSpeed"),
                            ("y", "tps"),
                        ],
                    )
            except Exception as ex:
                logging.error(f"Failed to export prefill vs tps markdown table: {ex}")
            logging.debug("Scatter plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving scatter plot: {e}")

        plt.close(fig)


def make_prefill_vs_tps_scatter_plot_llama31_8b(df_data: pd.DataFrame, save_path: str):
    """Create scatter plot for Llama 3.1 8B model only"""
    make_prefill_vs_tps_scatter_plot(df_data, save_path, model_filter="Llama 3.1 8B")


def make_prefill_vs_tps_scatter_plot_llama32_3b(df_data: pd.DataFrame, save_path: str):
    """Create scatter plot for Llama 3.2 3B model only"""
    make_prefill_vs_tps_scatter_plot(df_data, save_path, model_filter="Llama 3.2 3B")


def make_prefill_speed_by_quant_plots(df_data: pd.DataFrame, save_path_dir: str):
    """Create diamond point plots of prefill speed by quantization, per smartphone"""
    if df_data is None or df_data.empty:
        logging.debug("No performance data provided for stacked prefill plots.")
        return

    df = df_data.copy()
    if "model" not in df.columns or "prefillSpeed" not in df.columns:
        logging.debug("Required columns missing for stacked prefill plots.")
        return

    df["quant"] = (
        df["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )

    family_order = ["Llama 3.1 8B", "Llama 3.2 3B"]
    df = df[df["family"].isin(family_order)]
    if df.empty:
        logging.debug(
            "No matching rows for requested families in stacked prefill plots."
        )
        return

    df = df[df["inputTokens"] != 256]
    if df.empty:
        logging.debug("No data remaining after excluding 256 tokens.")
        return

    family_styles = {
        "Llama 3.1 8B": {"hatch": None, "edgecolor": "#1f77b4", "alpha": 0.95},
        "Llama 3.2 3B": {"hatch": None, "edgecolor": "#2ca02c", "alpha": 1.0},
    }

    grouped = (
        df.groupby(["deviceModel", "family", "quant", "inputTokens"])["prefillSpeed"]
        .mean()
        .reset_index()
    )

    unique_tokens = sorted(grouped["inputTokens"].unique())
    token_to_color = {
        128: "#1f77b4",
        64: "#ff7f0e",
        32: "#2ca02c"
    }

    for tok in unique_tokens:
        if tok not in token_to_color:
            token_to_color[tok] = "gray"

    devices = grouped["deviceModel"].unique()

    for device in devices:
        device_df = grouped[grouped["deviceModel"] == device]
        if device_df.empty:
            continue

        pivot = device_df.pivot_table(
            index=["quant", "family"],
            columns="inputTokens",
            values="prefillSpeed",
            aggfunc="mean",
            fill_value=0,
        )

        def _sort_key_with_family(q):
            for fam in family_order:
                if (q, fam) in pivot.index:
                    return quant_engine_group_and_order(q, fam)
            return quant_engine_group_and_order(q)

        quants = sorted(
            pivot.index.get_level_values("quant").unique(), key=_sort_key_with_family
        )

        x = np.arange(len(quants))
        bar_width = 0.38
        offsets = {
            family_order[0]: -bar_width / 2 - 0.02,
            family_order[1]: bar_width / 2 + 0.02,
        }

        with load_theme(THEME_NAME).set_font(family="sans-serif"):
            fig, ax = plt.subplots(figsize=(10, 16))

            marker_size = 100

            for fam in family_order:
                for tok in unique_tokens:
                    x_positions = []
                    y_positions = []
                    colors = []
                    edge_colors = []

                    for i, q in enumerate(quants):
                        try:
                            current_value = (
                                float(pivot.loc[(q, fam)].get(tok, 0))
                                if (q, fam) in pivot.index
                                else 0.0
                            )
                        except KeyError:
                            current_value = 0.0

                        x_pos = current_value

                        if x_pos > 0:
                            x_positions.append(x_pos)
                            y_positions.append(x[i] + offsets[fam])

                            point_color = token_to_color[tok]
                            point_edge_color = "black"

                            colors.append(point_color)
                            edge_colors.append(point_edge_color)

                    if x_positions:
                        marker_type = "8" if fam == family_order[1] else "D"
                        current_marker_size = marker_size * 1.5 if marker_type == "8" else marker_size
                        ax.scatter(
                            x_positions,
                            y_positions,
                            s=current_marker_size,
                            c=colors,
                            marker=marker_type,
                            edgecolors=edge_colors,
                            linewidths=1.0,
                            alpha=1.0
                            if fam == family_order[1]
                            else family_styles[fam]["alpha"],
                            label=str(tok) if fam == family_order[0] else None,
                        )

            ax.set_ylabel("Kwantyzacja", fontsize=16)
            ax.set_xlabel("Prędkość wstępnego przetwarzania (tokeny/s)", fontsize=16)
            ax.set_yticks(x)
            labeled_quants = [prepend_engine_symbol_to_quant(q) for q in quants]
            ax.set_yticklabels(labeled_quants, ha="right")
            ax.grid(True, linestyle="--", alpha=0.6, axis="x")
            ax.ticklabel_format(style="plain", axis="x")

            stack_legend = ax.legend(
                title="Wejściowe tokeny", loc="lower right", fontsize=10
            )

            rep_color = (
                token_to_color[
                    unique_tokens[min(len(unique_tokens) // 2, len(unique_tokens) - 1)]
                ]
                if unique_tokens
                else "gray"
            )

            family_handles = []
            for f in family_order:
                facecolor = rep_color
                edgecolor = "black"
                alpha = 0.95 if f == family_order[0] else 1.0
                marker_type = "D" if f == family_order[0] else "8"
                legend_marker_size = marker_size * 1.5 if marker_type == "8" else marker_size

                handle = ax.scatter(
                    [],
                    [],
                    s=legend_marker_size,
                    c=facecolor,
                    marker=marker_type,
                    edgecolors=edgecolor,
                    linewidths=1.0,
                    alpha=alpha,
                    label=f,
                )
                family_handles.append(handle)

            family_legend = ax.legend(
                handles=family_handles,
                title="Model",
                loc="lower right",
                bbox_to_anchor=(1.0, 0.08),
                fontsize=10,
            )
            ax.add_artist(stack_legend)
            ax.add_artist(family_legend)

            plt.tight_layout()
            plt.show()

            filename = f"stacked_prefill_by_quant__{device}.svg"
            save_path = os.path.join(save_path_dir, filename)
            try:
                logging.debug(f"Saving stacked prefill plot to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for fam in family_order:
                        for tok in unique_tokens:
                            for i, q in enumerate(quants):
                                try:
                                    current_value = (
                                        float(pivot.loc[(q, fam)].get(tok, 0))
                                        if (q, fam) in pivot.index
                                        else 0.0
                                    )
                                except KeyError:
                                    current_value = 0.0
                                plotted_width = current_value
                                if plotted_width > 0:
                                    export_rows.append(
                                        {
                                            "quant": q,
                                            "family": fam,
                                            "inputTokens": tok,
                                            "plottedWidth": plotted_width,
                                        }
                                    )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("quant", "quant"),
                                ("family", "family"),
                                ("inputTokens", "inputTokens"),
                                ("plottedWidth", "plottedWidth"),
                            ],
                        )
                except Exception as ex:
                    logging.error(
                        f"Failed to export stacked prefill markdown table for {device}: {ex}"
                    )
                logging.debug("Stacked prefill plot saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving stacked prefill plot for {device}: {e}")
            finally:
                plt.close(fig)


def make_ram_usage_by_quant_plots(df_data: pd.DataFrame, save_path_dir: str):
    df_filtered = df_data[
        df_data["application"].isin(["llama.cpp", "ExecuTorch"])
    ].copy()
    if df_filtered.empty:
        logging.debug(
            "No rows after filtering to applications llama.cpp and ExecuTorch"
        )
        return

    device_order = ["CPH2581", "IN2013", "SM-A256B"]
    df_filtered = df_filtered[df_filtered["deviceModel"].isin(device_order)]
    df_filtered = df_filtered[df_filtered["inputTokens"] == 64]
    if df_filtered.empty:
        logging.debug("No rows for selected devices or inputTokens=64")
        return

    df_filtered["quant"] = (
        df_filtered["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )

    grouped = (
        df_filtered.groupby(["family", "quant", "deviceModel"])
        .agg({"ramAvg": "mean", "ramMax": "max"})
        .reset_index()
    )

    device_colors = {
        "CPH2581": "#1f77b4",  # blue
        "IN2013": "#2ca02c",  # green
        "SM-A256B": "#ff7f0e",  # orange
    }
    device_name_map = {
        "CPH2581": "OnePlus 12",
        "IN2013": "OnePlus 8",
        "SM-A256B": "Samsung A25",
    }

    families = ["Llama 3.1 8B", "Llama 3.2 3B"]
    for fam in families:
        fam_df = grouped[grouped["family"] == fam]
        if fam_df.empty:
            logging.debug(f"No data for family {fam}")
            continue

        quants = sorted(
            fam_df["quant"].unique(), key=lambda q: quant_engine_group_and_order(q, fam)
        )
        if not quants:
            logging.debug(f"No quantizations for family {fam}")
            continue

        x = np.arange(len(quants))
        bar_width = 0.22
        offsets = {
            device_order[0]: -bar_width,
            device_order[1]: 0.0,
            device_order[2]: +bar_width,
        }

        with load_theme(THEME_NAME).set_font(family="sans-serif"):
            fig, ax = plt.subplots(figsize=(10, 16))

            for device in device_order:
                dev_data = fam_df[fam_df["deviceModel"] == device]
                widths = []
                avg_lines = []
                for q in quants:
                    row = dev_data[dev_data["quant"] == q]
                    if not row.empty:
                        widths.append(float(row["ramMax"].iloc[0]))
                        avg_lines.append(float(row["ramAvg"].iloc[0]))
                    else:
                        widths.append(0.0)
                        avg_lines.append(0.0)

                centers = x + offsets[device]
                ax.barh(
                    centers,
                    widths,
                    height=bar_width,
                    color=device_colors[device],
                    alpha=0.85,
                    label=device_name_map.get(device, device),
                    edgecolor="black",
                    linewidth=0.6,
                )
                half_h = bar_width / 2.0
                for cy, avg_val in zip(centers, avg_lines):
                    ax.vlines(
                        avg_val,
                        cy - half_h,
                        cy + half_h,
                        colors="white",
                        linewidth=2.2,
                        zorder=4,
                    )

            ax.set_ylabel("Kwantyzacja", fontsize=16)
            ax.set_xlabel("Zużycie RAM (MB)", fontsize=16)
            ax.set_yticks(x)
            ax.set_yticklabels(
                [prepend_engine_symbol_to_quant(q, fam) for q in quants], ha="right"
            )
            ax.grid(True, linestyle="--", alpha=0.6, axis="x")

            ax.legend(title="Smartfon", loc="lower right", fontsize=10)

            plt.tight_layout()
            plt.show()

            family_clean = fam.replace(" ", "_").replace(".", "")
            filename = f"ram_by_quant_{family_clean}__llama.cpp_ExecuTorch.svg"
            save_path = os.path.join(save_path_dir, filename)
            try:
                logging.debug(f"Saving RAM plot to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for device in device_order:
                        dev_data = fam_df[fam_df["deviceModel"] == device]
                        for q in quants:
                            row = dev_data[dev_data["quant"] == q]
                            if not row.empty:
                                export_rows.append(
                                    {
                                        "family": fam,
                                        "deviceModel": device,
                                        "quant": q,
                                        "ramMax": float(row["ramMax"].iloc[0]),
                                        "ramAvg": float(row["ramAvg"].iloc[0]),
                                    }
                                )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("family", "family"),
                                ("deviceModel", "deviceModel"),
                                ("quant", "quant"),
                                ("ramMax", "ramMax"),
                                ("ramAvg", "ramAvg"),
                            ],
                        )
                except Exception as ex:
                    logging.error(f"Failed to export RAM markdown table: {ex}")
                logging.debug("RAM plot saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving RAM plot: {e}")
            finally:
                plt.close(fig)


def make_power_draw_by_quant_plots(df_data: pd.DataFrame, save_path_dir: str):
    df = df_data[df_data["application"].isin(["llama.cpp", "ExecuTorch", "MLC"])].copy()
    if df.empty:
        logging.debug(
            "No rows after filtering to applications llama.cpp, ExecuTorch, and MLC for power plot"
        )
        return

    device_order = ["CPH2581", "IN2013", "SM-A256B"]
    df = df[df["deviceModel"].isin(device_order)]
    df = df[df["inputTokens"] == 64]
    if df.empty:
        logging.debug("No rows for selected devices or inputTokens=64 for power plot")
        return

    required_cols = [
        "model",
        "family",
        "deviceModel",
        "inferenceTime",
        "batInfo_usedCapacityMah_bef",
        "batInfo_usedCapacityMah_aft",
    ]
    for col in required_cols:
        if col not in df.columns:
            logging.debug(f"Required column missing for power plot: {col}")
            return

    df["quant"] = (
        df["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )

    df = df.copy()
    df["delta_mAh"] = df["batInfo_usedCapacityMah_aft"].astype(float) - df[
        "batInfo_usedCapacityMah_bef"
    ].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["delta_mAh", "inferenceTime"])
    df = df[df["delta_mAh"] > 0]
    df["timeSec"] = df["inferenceTime"].astype(float) / 1000.0
    df = df[df["timeSec"] > 0]
    if df.empty:
        logging.debug("No rows with valid battery deltas and time for power plot")
        return
    df["mA"] = df["delta_mAh"] * 3600.0 / df["timeSec"]

    grouped = df.groupby(["family", "quant", "deviceModel"])["mA"].mean().reset_index()

    total_current_grouped = (
        df.groupby(["family", "quant", "deviceModel"])["delta_mAh"].mean().reset_index()
    )

    device_colors = {
        "CPH2581": "#1f77b4",  # blue
        "IN2013": "#2ca02c",  # green
        "SM-A256B": "#ff7f0e",  # orange
    }
    device_name_map = {
        "CPH2581": "OnePlus 12",
        "IN2013": "OnePlus 8",
        "SM-A256B": "Samsung A25",
    }

    families = ["Llama 3.1 8B", "Llama 3.2 3B"]
    for fam in families:
        fam_df = grouped[grouped["family"] == fam]
        if fam_df.empty:
            logging.debug(f"No data for family {fam} in power plot")
            continue

        quants = sorted(
            fam_df["quant"].unique(), key=lambda q: quant_engine_group_and_order(q, fam)
        )
        if not quants:
            logging.debug(f"No quantizations for family {fam} in power plot")
            continue

        x = np.arange(len(quants))
        bar_width = 0.22
        offsets = {
            device_order[0]: -bar_width,
            device_order[1]: 0.0,
            device_order[2]: +bar_width,
        }

        with load_theme(THEME_NAME).set_font(family="sans-serif"):
            fig, ax = plt.subplots(figsize=(10, 16))

            for device in device_order:
                dev_data = fam_df[fam_df["deviceModel"] == device]
                widths = []
                for q in quants:
                    row = dev_data[dev_data["quant"] == q]
                    if not row.empty:
                        widths.append(float(row["mA"].iloc[0]))
                    else:
                        widths.append(0.0)

                centers = x + offsets[device]
                ax.barh(
                    centers,
                    widths,
                    height=bar_width,
                    color=device_colors.get(device, "gray"),
                    alpha=0.85,
                    label=device_name_map.get(device, device),
                    edgecolor="black",
                    linewidth=0.6,
                )

            ax2 = ax.twiny()
            ax2.set_xlabel(
                "Całkowite zużycie prądu podczas inferencji (mAh)", fontsize=16
            )

            fam_total_current = total_current_grouped[
                total_current_grouped["family"] == fam
            ]
            all_total_currents = []
            for device in device_order:
                dev_total_data = fam_total_current[
                    fam_total_current["deviceModel"] == device
                ]
                for q in quants:
                    row = dev_total_data[dev_total_data["quant"] == q]
                    if not row.empty:
                        all_total_currents.append(float(row["delta_mAh"].iloc[0]))

            if all_total_currents:
                min_total = min(all_total_currents)
                max_total = max(all_total_currents)
                ax2.set_xlim(min_total * 0.9, max_total * 1.1)

            ax2.grid(True, linestyle=":", alpha=0.4, axis="x")

            for device in device_order:
                dev_total_data = fam_total_current[
                    fam_total_current["deviceModel"] == device
                ]
                total_currents = []
                for q in quants:
                    row = dev_total_data[dev_total_data["quant"] == q]
                    if not row.empty:
                        total_currents.append(float(row["delta_mAh"].iloc[0]))
                    else:
                        total_currents.append(0.0)

                centers = x + offsets[device]
                ax2.scatter(
                    total_currents,
                    centers,
                    marker="D",
                    s=100,
                    color=device_colors.get(device, "gray"),
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=10,
                )

            ax.set_ylabel("Kwantyzacja", fontsize=16)
            ax.set_xlabel("Pobór mocy w ciągu godziny (mA)", fontsize=16)
            ax.set_yticks(x)
            ax.set_yticklabels(
                [prepend_engine_symbol_to_quant(q, fam) for q in quants], ha="right"
            )
            ax.grid(True, linestyle="--", alpha=0.6, axis="x")

            device_legends = [
                mpatches.Patch(
                    color=device_colors.get(device, "gray"),
                    label=device_name_map.get(device, device),
                )
                for device in device_order
            ]

            diamond_legend = Line2D(
                [0],
                [0],
                marker="D",
                color="gray",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                linestyle="None",
                label="Całkowite zużycie (mAh)",
            )

            all_legends = device_legends + [diamond_legend]
            ax.legend(
                handles=all_legends,
                title="Smartfon / Typ pomiaru",
                loc="lower right",
                fontsize=10,
            )

            plt.tight_layout()
            plt.show()

            family_clean = fam.replace(" ", "_").replace(".", "")
            filename = f"power_by_quant_{family_clean}__llama.cpp_ExecuTorch_MLC.svg"
            save_path = os.path.join(save_path_dir, filename)
            try:
                logging.debug(f"Saving power plot to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for device in device_order:
                        dev_data = fam_df[fam_df["deviceModel"] == device]
                        for q in quants:
                            row = dev_data[dev_data["quant"] == q]
                            if not row.empty:
                                export_rows.append(
                                    {
                                        "family": fam,
                                        "deviceModel": device,
                                        "quant": q,
                                        "mA": float(row["mA"].iloc[0]),
                                    }
                                )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("family", "family"),
                                ("deviceModel", "deviceModel"),
                                ("quant", "quant"),
                                ("mA", "mA"),
                            ],
                        )
                except Exception as ex:
                    logging.error(f"Failed to export power markdown table: {ex}")
                logging.debug("Power plot saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving power plot: {e}")
            finally:
                plt.close(fig)


def make_temperature_heatmap(
    df_data: pd.DataFrame, device_model: str, save_path: Optional[str] = None
):
    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        filtered_data = df_data[df_data["deviceModel"] == device_model].copy()
        if filtered_data.empty:
            logging.debug(f"No data for device: {device_model}")
            return None, None

        filtered_data = filtered_data[filtered_data["inputTokens"] == 64]
        filtered_data["quant"] = filtered_data["model"].str.split().str[-1]

        families = ["Llama 3.1 8B", "Llama 3.2 3B"]

        sensors = [
            "sensTemp_shell_front_aft",
            "sensTemp_shell_frame_aft",
            "sensTemp_shell_back_aft",
        ]
        sensor_labels = ["Przednia obudowa", "Rama obudowy", "Tylna obudowa"]

        def _sort_key_with_family(q):
            for fam in families:
                if not filtered_data[
                    (filtered_data["quant"] == q) & (filtered_data["family"] == fam)
                ].empty:
                    return quant_engine_group_and_order(q, fam)
            return quant_engine_group_and_order(q)

        quants = sorted(filtered_data["quant"].unique(), key=_sort_key_with_family)

        grouped = (
            filtered_data.groupby(["quant", "family"])[sensors].max().reset_index()
        )

        all_temps = grouped[sensors].values.flatten()
        min_temp = np.nanmin(all_temps)
        max_temp = np.nanmax(all_temps)
        norm = Normalize(vmin=min_temp, vmax=max_temp)
        cmap = colormaps["Reds"]

        fig, ax = plt.subplots(figsize=(len(quants) * 1.2, len(sensors) * 2))

        ax.set_xlim(0, len(quants))
        ax.set_ylim(0, len(sensors))

        ax.set_xticks(np.arange(len(quants)) + 0.5)
        ax.set_xticklabels(
            [prepend_engine_symbol_to_quant(q) for q in quants], rotation=45, ha="right"
        )
        ax.set_yticks(np.arange(len(sensors)) + 0.5)
        ax.set_yticklabels(sensor_labels)

        for xi in range(len(quants) + 1):
            ax.axvline(xi, color="black", lw=1)
        for yi in range(len(sensors) + 1):
            ax.axhline(yi, color="black", lw=1)

        def get_text_color(bg_color):
            r, g, b = bg_color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "white" if luminance < 0.5 else "black"

        for i, q in enumerate(quants):
            for j, sensor in enumerate(sensors):
                cell_x = i
                cell_y = j
                for fam_idx, fam in enumerate(families):
                    row = grouped[(grouped["quant"] == q) & (grouped["family"] == fam)]
                    if row.empty:
                        temp_val = np.nan
                    else:
                        temp_val = row[sensor].to_numpy().max()

                    if np.isnan(temp_val):
                        color = "white"
                    else:
                        color = cmap(norm(temp_val))

                    if fam == "Llama 3.1 8B":
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y + 1),
                                (cell_x, cell_y + 1),
                            ]
                        )
                    else:
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y),
                                (cell_x + 1, cell_y + 1),
                            ]
                        )

                    poly = Polygon(points, facecolor=color, edgecolor="black", lw=0.5)
                    ax.add_patch(poly)

                    if not np.isnan(temp_val):
                        center = np.mean(points, axis=0)
                        text_color = get_text_color(color)
                        ax.text(
                            center[0],
                            center[1],
                            f"{temp_val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            color=text_color,
                        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=0.02,
            pad=0.01,
            label="Temperatura (°C)",
        )
        cbar.set_label("Temperatura (°C)", fontsize=16)

        ax.set_xlabel("Kwantyzacja", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)

        plt.tight_layout()
        plt.show()

        if save_path:
            try:
                logging.debug(f"Saving heatmap to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for q in quants:
                        for fam in families:
                            row = grouped[
                                (grouped["quant"] == q) & (grouped["family"] == fam)
                            ]
                            if row.empty:
                                continue
                            for sensor in sensors:
                                val = row[sensor].to_numpy().max()
                                if pd.notna(val):
                                    export_rows.append(
                                        {
                                            "quant": q,
                                            "family": fam,
                                            "sensor": sensor,
                                            "maxTempC": float(val),
                                        }
                                    )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("quant", "quant"),
                                ("family", "family"),
                                ("sensor", "sensor"),
                                ("maxTempC", "maxTempC"),
                            ],
                        )
                except Exception as ex:
                    logging.error(
                        f"Failed to export shell heatmap markdown table: {ex}"
                    )
                logging.debug("Heatmap saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving heatmap: {e}")
            finally:
                plt.close(fig)

        return fig, ax


def make_cpu_temperature_heatmap(
    df_data: pd.DataFrame, device_model: str, save_path: Optional[str] = None
):
    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        filtered_data = df_data[df_data["deviceModel"] == device_model].copy()
        if filtered_data.empty:
            logging.debug(f"No data for device: {device_model}")
            return None, None

        filtered_data = filtered_data[filtered_data["inputTokens"] == 64]
        filtered_data["quant"] = filtered_data["model"].str.split().str[-1]

        families = ["Llama 3.1 8B", "Llama 3.2 3B"]

        if device_model == "IN2013":  # OnePlus 8
            sensor_keys = [
                "cpu-0-0-usr",
                "cpu-0-1-usr",
                "cpu-0-2-usr",
                "cpu-0-3-usr",
                "cpuss-0-usr",
                "cpuss-1-usr",
                "cpu-1-0-usr",
                "cpu-1-1-usr",
                "cpu-1-2-usr",
                "cpu-1-3-usr",
                "cpu-1-4-usr",
                "cpu-1-5-usr",
                "cpu-1-6-usr",
                "cpu-1-7-usr",
            ]
        elif device_model == "CPH2581":  # OnePlus 12
            sensor_keys = [
                "cpuss-0",
                "cpuss-1",
                "cpuss-2",
                "cpuss-3",
                "cpu-2-0-0",
                "cpu-2-0-1",
                "cpu-2-1-0",
                "cpu-2-1-1",
                "cpu-2-2-0",
                "cpu-2-2-1",
                "cpu-1-0-0",
                "cpu-1-0-1",
                "cpu-1-0-1",
                "cpu-1-1-0",
                "cpu-1-1-1",
            ]
        else:
            logging.debug(f"Unknown device: {device_model}")
            return None, None

        sensors = [f"sensTemp_{key}_aft" for key in sensor_keys]
        sensor_labels = [
            key for key in sensor_keys
        ]

        def _sort_key_with_family(q):
            for fam in families:
                if not filtered_data[
                    (filtered_data["quant"] == q) & (filtered_data["family"] == fam)
                ].empty:
                    return quant_engine_group_and_order(q, fam)
            return quant_engine_group_and_order(q)

        quants = sorted(filtered_data["quant"].unique(), key=_sort_key_with_family)

        grouped = (
            filtered_data.groupby(["quant", "family"])[sensors].max().reset_index()
        )

        all_temps = grouped[sensors].values.flatten()
        min_temp = np.nanmin(all_temps)
        max_temp = np.nanmax(all_temps)
        norm = Normalize(vmin=min_temp, vmax=max_temp)
        cmap = colormaps["Reds"]

        fig, ax = plt.subplots(figsize=(len(quants) * 1.2, len(sensors) * 1.5))

        ax.set_xlim(0, len(quants))
        ax.set_ylim(0, len(sensors))

        ax.set_xticks(np.arange(len(quants)) + 0.5)
        ax.set_xticklabels(
            [prepend_engine_symbol_to_quant(q) for q in quants], rotation=45, ha="right"
        )
        ax.set_yticks(np.arange(len(sensors)) + 0.5)
        ax.set_yticklabels(sensor_labels)

        for xi in range(len(quants) + 1):
            ax.axvline(xi, color="black", lw=1)
        for yi in range(len(sensors) + 1):
            ax.axhline(yi, color="black", lw=1)

        def get_text_color(bg_color):
            r, g, b = bg_color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "white" if luminance < 0.5 else "black"

        for i, q in enumerate(quants):
            for j, sensor in enumerate(sensors):
                cell_x = i
                cell_y = j
                for fam_idx, fam in enumerate(families):
                    row = grouped[(grouped["quant"] == q) & (grouped["family"] == fam)]
                    if row.empty:
                        temp_val = np.nan
                    else:
                        temp_val = row[sensor].to_numpy().max()

                    if np.isnan(temp_val):
                        color = "white"
                    else:
                        color = cmap(norm(temp_val))

                    if fam == "Llama 3.1 8B":
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y + 1),
                                (cell_x, cell_y + 1),
                            ]
                        )
                    else:
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y),
                                (cell_x + 1, cell_y + 1),
                            ]
                        )

                    poly = Polygon(points, facecolor=color, edgecolor="black", lw=0.5)
                    ax.add_patch(poly)

                    if not np.isnan(temp_val):
                        center = np.mean(points, axis=0)
                        text_color = get_text_color(color)
                        ax.text(
                            center[0],
                            center[1],
                            f"{temp_val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            color=text_color,
                        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=0.02,
            pad=0.01,
            label="Temperatura (°C)",
        )
        cbar.set_label("Temperatura (°C)", fontsize=16)

        ax.set_xlabel("Kwantyzacja", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)

        plt.tight_layout()
        plt.show()

        if save_path:
            try:
                logging.debug(f"Saving CPU heatmap to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for q in quants:
                        for fam in families:
                            row = grouped[
                                (grouped["quant"] == q) & (grouped["family"] == fam)
                            ]
                            if row.empty:
                                continue
                            for sensor in sensors:
                                val = row[sensor].to_numpy().max()
                                if pd.notna(val):
                                    export_rows.append(
                                        {
                                            "quant": q,
                                            "family": fam,
                                            "sensor": sensor,
                                            "maxTempC": float(val),
                                        }
                                    )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("quant", "quant"),
                                ("family", "family"),
                                ("sensor", "sensor"),
                                ("maxTempC", "maxTempC"),
                            ],
                        )
                except Exception as ex:
                    logging.error(f"Failed to export CPU heatmap markdown table: {ex}")
                logging.debug("CPU Heatmap saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving CPU heatmap: {e}")
            finally:
                plt.close(fig)

        return fig, ax


def make_gpu_temperature_heatmap(
    df_data: pd.DataFrame, device_model: str, save_path: Optional[str] = None
):
    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        filtered_data = df_data[df_data["deviceModel"] == device_model].copy()
        if filtered_data.empty:
            logging.debug(f"No data for device: {device_model}")
            return None, None

        filtered_data = filtered_data[filtered_data["inputTokens"] == 64]
        filtered_data["quant"] = filtered_data["model"].str.split().str[-1]

        families = ["Llama 3.1 8B", "Llama 3.2 3B"]

        if device_model == "IN2013":  # OnePlus 8
            sensor_keys = ["gpuss-0-usr", "gpuss-1-usr", "gpuss-max-step"]
        elif device_model == "CPH2581":  # OnePlus 12
            sensor_keys = [
                "gpuss-0",
                "gpuss-1",
                "gpuss-2",
                "gpuss-3",
                "gpuss-4",
                "gpuss-5",
                "gpuss-6",
                "gpuss-7",
            ]
        else:
            logging.debug(f"Unknown device: {device_model}")
            return None, None

        sensors = [f"sensTemp_{key}_aft" for key in sensor_keys]
        sensor_labels = [key for key in sensor_keys]

        def _sort_key_with_family(q):
            for fam in families:
                if not filtered_data[
                    (filtered_data["quant"] == q) & (filtered_data["family"] == fam)
                ].empty:
                    return quant_engine_group_and_order(q, fam)
            return quant_engine_group_and_order(q)

        quants = sorted(filtered_data["quant"].unique(), key=_sort_key_with_family)

        grouped = (
            filtered_data.groupby(["quant", "family"])[sensors].max().reset_index()
        )

        all_temps = grouped[sensors].values.flatten()
        min_temp = np.nanmin(all_temps)
        max_temp = np.nanmax(all_temps)
        norm = Normalize(vmin=min_temp, vmax=max_temp)
        cmap = colormaps["Reds"]

        fig, ax = plt.subplots(figsize=(len(quants) * 1.2, len(sensors) * 1.5))

        ax.set_xlim(0, len(quants))
        ax.set_ylim(0, len(sensors))

        ax.set_xticks(np.arange(len(quants)) + 0.5)
        ax.set_xticklabels(
            [prepend_engine_symbol_to_quant(q) for q in quants], rotation=45, ha="right"
        )
        ax.set_yticks(np.arange(len(sensors)) + 0.5)
        ax.set_yticklabels(sensor_labels)

        for xi in range(len(quants) + 1):
            ax.axvline(xi, color="black", lw=1)
        for yi in range(len(sensors) + 1):
            ax.axhline(yi, color="black", lw=1)

        def get_text_color(bg_color):
            r, g, b = bg_color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "white" if luminance < 0.5 else "black"

        for i, q in enumerate(quants):
            for j, sensor in enumerate(sensors):
                cell_x = i
                cell_y = j
                for fam_idx, fam in enumerate(families):
                    row = grouped[(grouped["quant"] == q) & (grouped["family"] == fam)]
                    if row.empty:
                        temp_val = np.nan
                    else:
                        temp_val = row[sensor].to_numpy().max()

                    if np.isnan(temp_val):
                        color = "white"
                    else:
                        color = cmap(norm(temp_val))

                    if fam == "Llama 3.1 8B":
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y + 1),
                                (cell_x, cell_y + 1),
                            ]
                        )
                    else:
                        points = np.array(
                            [
                                (cell_x, cell_y),
                                (cell_x + 1, cell_y),
                                (cell_x + 1, cell_y + 1),
                            ]
                        )

                    poly = Polygon(points, facecolor=color, edgecolor="black", lw=0.5)
                    ax.add_patch(poly)

                    if not np.isnan(temp_val):
                        center = np.mean(points, axis=0)
                        text_color = get_text_color(color)
                        ax.text(
                            center[0],
                            center[1],
                            f"{temp_val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            color=text_color,
                        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=0.02,
            pad=0.01,
            label="Temperatura (°C)",
        )
        cbar.set_label("Temperatura (°C)", fontsize=16)

        ax.set_xlabel("Kwantyzacja", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)

        plt.tight_layout()
        plt.show()

        if save_path:
            try:
                logging.debug(f"Saving GPU heatmap to {save_path}...")
                fig.savefig(save_path, format="svg", bbox_inches="tight")
                try:
                    export_rows = []
                    for q in quants:
                        for fam in families:
                            row = grouped[
                                (grouped["quant"] == q) & (grouped["family"] == fam)
                            ]
                            if row.empty:
                                continue
                            for sensor in sensors:
                                val = row[sensor].to_numpy().max()
                                if pd.notna(val):
                                    export_rows.append(
                                        {
                                            "quant": q,
                                            "family": fam,
                                            "sensor": sensor,
                                            "maxTempC": float(val),
                                        }
                                    )
                    if export_rows:
                        df_export = pd.DataFrame(export_rows)
                        _save_markdown_table_for_plot(
                            save_path,
                            [{"df": df_export}],
                            [
                                ("quant", "quant"),
                                ("family", "family"),
                                ("sensor", "sensor"),
                                ("maxTempC", "maxTempC"),
                            ],
                        )
                except Exception as ex:
                    logging.error(f"Failed to export GPU heatmap markdown table: {ex}")
                logging.debug("GPU Heatmap saved successfully.")
            except Exception as e:
                logging.debug(f"Error saving GPU heatmap: {e}")
            finally:
                plt.close(fig)

        return fig, ax


def make_perf_plots(df_data: pd.DataFrame, save_path_dir: str):
    # RAM usage vs quantization plots
    make_ram_usage_by_quant_plots(df_data, save_path_dir)
    # power draw vs quantization plotS
    make_power_draw_by_quant_plots(df_data, save_path_dir)
    # TPS plots
    make_model_based_tps_plots(df_data, save_path_dir)

    # Prefill speed vs TPS scatter plots
    make_prefill_vs_tps_scatter_plot_llama31_8b(
        df_data,
        os.path.join(save_path_dir, "scatter_plot_prefill_vs_tps_Llama_3.1_8B.svg"),
    )
    make_prefill_vs_tps_scatter_plot_llama32_3b(
        df_data,
        os.path.join(save_path_dir, "scatter_plot_prefill_vs_tps_Llama_3.2_3B.svg"),
    )

    # Prefill speed by quantization plots
    make_prefill_speed_by_quant_plots(df_data, save_path_dir)

    # Temperature heatmaps per smartphone
    make_temperature_heatmap(
        df_data,
        "CPH2581",
        os.path.join(save_path_dir, "temp_heatmap_shell_oneplus12.svg"),
    )
    make_temperature_heatmap(
        df_data,
        "IN2013",
        os.path.join(save_path_dir, "temp_heatmap_shell_oneplus8.svg"),
    )

    make_cpu_temperature_heatmap(
        df_data,
        "CPH2581",
        os.path.join(save_path_dir, "temp_heatmap_cpu_oneplus12.svg"),
    )
    make_cpu_temperature_heatmap(
        df_data, "IN2013", os.path.join(save_path_dir, "temp_heatmap_cpu_oneplus8.svg")
    )

    make_gpu_temperature_heatmap(
        df_data,
        "CPH2581",
        os.path.join(save_path_dir, "temp_heatmap_gpu_oneplus12.svg"),
    )
    make_gpu_temperature_heatmap(
        df_data, "IN2013", os.path.join(save_path_dir, "temp_heatmap_gpu_oneplus8.svg")
    )


def _draw_accuracy_vs_time_plot(
    datasets: list[dict], x_title: str, y_title: str, save_path: Optional[str] = None
):
    with load_theme(THEME_NAME).set_font(family="sans-serif"):
        all_x = np.concatenate([d["df"]["x"].values for d in datasets])
        all_x_sorted = np.sort(np.unique(all_x))
        use_broken = False
        ax2 = None
        left_max = None
        if len(all_x_sorted) >= 2:
            range_x = all_x_sorted[-1] - all_x_sorted[0]
            if range_x == 0:
                range_x = 1
            diffs = np.diff(all_x_sorted)
            max_diff_idx = np.argmax(diffs)
            if diffs[max_diff_idx] > 0.2 * range_x and len(all_x_sorted) > 2:
                use_broken = True
                split_idx = max_diff_idx + 1
                left_max = all_x_sorted[split_idx - 1]
                _right_min = all_x_sorted[split_idx]
        if use_broken:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(12, 7), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
            )
            fig.subplots_adjust(wspace=0.05)
            ax1.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.tick_params(labelleft=False)
        else:
            fig, ax1 = plt.subplots(figsize=(12, 7))
        draggable_texts = []

        if datasets:
            all_accs = np.concatenate([d["df"]["y"].values for d in datasets])
            y_range = float(all_accs.max() - all_accs.min()) if len(all_accs) else 0.0
        else:
            y_range = 0.0
        threshold = y_range * 0.1 if y_range > 0 else 0.05

        for i, data in enumerate(datasets):
            df = data["df"]
            color = data.get("color", f"C{i}")
            name = data.get("name", f"Dataset {i + 1}")
            pareto_quants = set()
            xs_pf: list[float] = []
            ys_pf: list[float] = []
            try:
                if {"x", "y", "quant"}.issubset(df.columns):
                    df_ann = df[["x", "y", "quant"]].dropna().copy()
                    if not df_ann.empty:
                        idxs = df_ann.groupby("x")["y"].idxmax()
                        df_unique = df_ann.loc[idxs].sort_values(by="x", ascending=True)
                        current_max_y = -np.inf
                        for _, r in df_unique.iterrows():
                            x_val = float(r["x"])
                            y_val = float(r["y"])
                            if y_val >= current_max_y - 1e-12:
                                pareto_quants.add(str(r["quant"]))
                                xs_pf.append(x_val)
                                ys_pf.append(y_val)
                                current_max_y = y_val
            except Exception as e:
                logging.debug(f"Pareto membership (accuracy vs time) failed: {e}")
            if use_broken:
                left_df = df[df["x"] <= left_max]
                right_df = df[df["x"] > left_max]
                label_on_left = not left_df.empty
                if not left_df.empty:
                    ax1.scatter(
                        left_df["x"],
                        left_df["y"],
                        s=80,
                        alpha=0.7,
                        color=color,
                        label=name if label_on_left else None,
                        zorder=2,
                    )
                if not right_df.empty:
                    ax2.scatter(
                        right_df["x"],
                        right_df["y"],
                        s=80,
                        alpha=0.7,
                        color=color,
                        label=name if not label_on_left else None,
                        zorder=2,
                    )
            else:
                ax1.scatter(
                    df["x"], df["y"], s=80, alpha=0.7, color=color, label=name, zorder=2
                )
            for _, row in df.iterrows():
                point_ax = ax1 if not use_broken or row["x"] <= left_max else ax2
                text = point_ax.text(
                    x=row["x"] + (0.02 if row["x"] >= 0 else 0.01),
                    y=row["y"] + 0.01,
                    s=row["quant"],
                    fontsize=11,
                    fontweight=(
                        "bold" if str(row["quant"]) in pareto_quants else "normal"
                    ),
                    rotation=45,
                    rotation_mode="anchor",
                    ha="right",
                    va="bottom",
                    color=color,
                    zorder=100,
                )
                dt = DraggableText(text, row["x"], row["y"], threshold, point_ax, color)
                draggable_texts.append(dt)
                dt.update_line()
            try:
                if len(xs_pf) >= 2:
                    if use_broken:
                        left_pts = [
                            (x, y) for x, y in zip(xs_pf, ys_pf) if x <= left_max
                        ]
                        right_pts = [
                            (x, y) for x, y in zip(xs_pf, ys_pf) if x > left_max
                        ]
                        if len(left_pts) >= 2:
                            ax1.plot(
                                [p[0] for p in left_pts],
                                [p[1] for p in left_pts],
                                linestyle="--",
                                linewidth=1.0,
                                color=color,
                                alpha=0.9,
                                zorder=1,
                            )
                        if len(right_pts) >= 2:
                            ax2.plot(
                                [p[0] for p in right_pts],
                                [p[1] for p in right_pts],
                                linestyle="--",
                                linewidth=1.0,
                                color=color,
                                alpha=0.9,
                                zorder=1,
                            )
                    else:
                        ax1.plot(
                            xs_pf,
                            ys_pf,
                            linestyle="--",
                            linewidth=1.0,
                            color=color,
                            alpha=0.9,
                            zorder=1,
                        )
            except Exception as e:
                logging.debug(f"Pareto front line (accuracy vs time) failed: {e}")
        ax1.set_xlabel(x_title, fontsize=16)
        ax1.set_ylabel(y_title, fontsize=16)
        ax1.grid(True, alpha=0.3, linestyle="--", axis="both", which="both", zorder=0)
        if use_broken:
            ax2.grid(
                True, alpha=0.3, linestyle="--", axis="both", which="both", zorder=0
            )
        if datasets:
            all_x_vals = np.concatenate([d["df"]["x"].values for d in datasets])
            all_y = np.concatenate([d["df"]["y"].values for d in datasets])
            ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            ypad = (ymax - ymin) * 0.08 if ymax > ymin else 0.02
            ylim = (max(0.0, ymin - ypad), min(1.0, ymax + ypad + 0.02))
            ax1.set_ylim(ylim)
            ax1.ticklabel_format(axis="x", style="plain", useOffset=False)
            if use_broken:
                ax2.set_ylim(ylim)
                ax2.ticklabel_format(axis="x", style="plain", useOffset=False)
                left_x_vals = all_x_vals[all_x_vals <= left_max]
                right_x_vals = all_x_vals[all_x_vals > left_max]
                if len(left_x_vals) > 0:
                    lmin, lmax = (
                        float(np.nanmin(left_x_vals)),
                        float(np.nanmax(left_x_vals)),
                    )
                    lpad = (lmax - lmin) * 0.08 if lmax > lmin else 0.02
                    ax1.set_xlim(max(0.0, lmin - lpad), lmax + lpad)
                if len(right_x_vals) > 0:
                    rmin, rmax = (
                        float(np.nanmin(right_x_vals)),
                        float(np.nanmax(right_x_vals)),
                    )
                    current_range = rmax - rmin
                    rpad = (current_range * 0.08) if current_range > 0 else 0.02
                    rmin_padded = rmin - rpad
                    rmax_padded = rmax + rpad
                    padded_range = rmax_padded - rmin_padded
                    if padded_range < 10:
                        mid = (rmin_padded + rmax_padded) / 2
                        half = 5.0
                        rmin_padded = mid - half
                        rmax_padded = mid + half
                    rmin_rounded = math.floor(rmin_padded / 10) * 10
                    rmax_rounded = math.ceil(rmax_padded / 10) * 10
                    if rmax_rounded - rmin_rounded < 10:
                        rmax_rounded = rmin_rounded + 10
                    ax2.set_xlim(rmin_rounded, rmax_rounded)
        else:
            ax1.set_xlim(0.0, 1.0)
            ax1.set_ylim(0.0, 1.0)
        if len(datasets) > 1:
            ax1.legend(loc="best", fontsize=12)
        fig.tight_layout(pad=0.5)
    plt.show()
    if save_path:
        try:
            logging.debug(f"Saving plot to {save_path}...")
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            try:
                export_frames = []
                for d in datasets:
                    df = d.get("df")
                    if df is None or getattr(df, "empty", True):
                        continue
                    keep_cols = [c for c in ["x", "y", "quant"] if c in df.columns]
                    if not keep_cols:
                        continue
                    export_frames.append(df[keep_cols].dropna(subset=keep_cols))
                if export_frames:
                    combined = pd.concat(export_frames, axis=0, ignore_index=True)
                    _save_markdown_table_for_plot(
                        save_path,
                        [{"df": combined}],
                        [("x", "x"), ("y", "y"), ("quant", "quant")],
                    )
            except Exception as ex:
                logging.error(f"Failed to export accuracy vs time markdown table: {ex}")
            logging.debug("Plot saved successfully.")
        except Exception as e:
            logging.debug(f"Error saving plot: {e}")
        finally:
            plt.close(fig)


def _prepare_datasets_for_device(
    df_perf: pd.DataFrame,
    device_model: str,
    family_to_eval_entries: dict,
    metric_key: str,
    family_colors: dict,
):
    df = df_perf.copy()
    df = df[df["deviceModel"] == device_model]
    df = df[df["inputTokens"] == 64]
    if df.empty:
        return []

    df["quant"] = (
        df["model"]
        .astype(str)
        .apply(lambda m: m.split(" ")[-1] if isinstance(m, str) else m)
    )

    grouped = df.groupby(["family", "quant"])["inferenceTime"].mean().reset_index()
    grouped["timeSec"] = grouped["inferenceTime"].astype(float) / 1000.0

    datasets = []
    for fam, eval_entries in family_to_eval_entries.items():
        fam_df = grouped[grouped["family"] == fam].copy()
        if fam_df.empty:
            continue

        eval_map = {
            d["quant"]: d.get(metric_key)
            for d in eval_entries
            if d.get(metric_key) is not None
        }

        fam_df["acc"] = fam_df["quant"].map(eval_map)
        fam_df = fam_df.dropna(subset=["acc"])
        fam_df = fam_df[fam_df["quant"] != "f16"]

        if fam_df.empty:
            continue

        plot_df = pd.DataFrame(
            {
                "x": fam_df["timeSec"].astype(float),
                "y": fam_df["acc"].astype(float),
                "quant": fam_df["quant"].astype(str),
            }
        )

        datasets.append(
            {"df": plot_df, "color": family_colors.get(fam, None), "name": fam}
        )

    return datasets


def make_accuracy_vs_inference_time_plots_per_device(
    df_perf: pd.DataFrame,
    family_to_eval_entries: dict,
    metric_key: str,
    title_prefix: str,
    filename_prefix: str,
    save_path_dir: str,
):
    family_colors = {"Llama 3.1 8B": "blue", "Llama 3.2 3B": "green"}

    device_models = df_perf["deviceModel"].dropna().unique()
    for device_model in device_models:
        datasets = _prepare_datasets_for_device(
            df_perf=df_perf,
            device_model=device_model,
            family_to_eval_entries=family_to_eval_entries,
            metric_key=metric_key,
            family_colors=family_colors,
        )
        if not datasets:
            logging.debug(
                f"No data to plot for device {device_model} and metric {metric_key}"
            )
            continue

        filename = f"{filename_prefix}__{device_model}.svg"
        save_path = os.path.join(save_path_dir, filename)

        _draw_accuracy_vs_time_plot(
            datasets,
            x_title="Czas inferencji (s)",
            y_title="Dokładność",
            save_path=save_path,
        )


save_path_dir = os.path.join(".", "plots", str(int(datetime.now().timestamp())))
os.makedirs(save_path_dir, exist_ok=True)

llama31_8b_ifeval = load_eval_data("llama31_8b_ifeval")
llama31_8b_mmlu = load_eval_data("llama31_8b_mmlu")
llama32_3b_ifeval = load_eval_data("llama32_3b_ifeval")
llama32_3b_mmlu = load_eval_data("llama32_3b_mmlu")

mmlu_data: list[DataEntry] = [
    {"data": llama31_8b_mmlu, "name": "LLama3.1 8B", "color": "blue"},
    {"data": llama32_3b_mmlu, "name": "LLama3.2 3B", "color": "green"},
]

ifeval_data: list[DataEntry] = [
    {"data": llama31_8b_ifeval, "name": "LLama3.1 8B", "color": "blue"},
    {"data": llama32_3b_ifeval, "name": "LLama3.2 3B", "color": "green"},
]

make_mmlu_plot(mmlu_data, os.path.join(save_path_dir, "mmlu_plot.svg"))
make_ifeval_plot(ifeval_data, os.path.join(save_path_dir, "ifeval_plot.svg"))

make_mmlu_vs_ifeval_plot(
    mmlu_data,
    ifeval_data,
    os.path.join(save_path_dir, "mmlu_vs_ifeval_plot.svg"),
)

llama_offline_perf = load_excel_worksheet("./output2.xlsx", "llama_perf")
llama_offline_perf = llama_offline_perf[
    ~(
        (llama_offline_perf["model"].str.contains("BF16"))
        & (llama_offline_perf["application"] == "llama.cpp")
    )
]
llama_offline_perf = llama_offline_perf[~(llama_offline_perf["inputTokens"] == 512)]


make_perf_plots(llama_offline_perf, save_path_dir)


# Build mapping from family name used in perf data -> eval entries
def _select_eval_entries(entries: list[DataEntry], family_label: str) -> list[dict]:
    match_token = "3.1" if "3.1" in family_label else "3.2"
    for e in entries:
        if match_token in e["name"]:
            return e["data"]
    return []


family_to_mmlu_entries = {
    "Llama 3.1 8B": _select_eval_entries(mmlu_data, "Llama 3.1 8B"),
    "Llama 3.2 3B": _select_eval_entries(mmlu_data, "Llama 3.2 3B"),
}

family_to_ifeval_entries = {
    "Llama 3.1 8B": _select_eval_entries(ifeval_data, "Llama 3.1 8B"),
    "Llama 3.2 3B": _select_eval_entries(ifeval_data, "Llama 3.2 3B"),
}

# MMLU and IFEval vs inference time (s)
make_accuracy_vs_inference_time_plots_per_device(
    df_perf=llama_offline_perf,
    family_to_eval_entries=family_to_mmlu_entries,
    metric_key="mmlu_acc",
    title_prefix="MMLU vs czas inferencji",
    filename_prefix="mmlu_vs_inference_time",
    save_path_dir=save_path_dir,
)

make_accuracy_vs_inference_time_plots_per_device(
    df_perf=llama_offline_perf,
    family_to_eval_entries=family_to_ifeval_entries,
    metric_key="ifeval_inst_loose_acc",
    title_prefix="IFEval vs czas inferencji",
    filename_prefix="ifeval_vs_inference_time",
    save_path_dir=save_path_dir,
)
