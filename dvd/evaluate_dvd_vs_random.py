"""
Evaluation script comparing DVD agent vs Random Sampling for ADOS scoring.
Generates comparison plots for accuracy, F1 score, and regression error.
"""

import json
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

# Load data



# Style settings from reference
colors = ["#da81c1", "#7dbfa7", "#b0d766", "#8ca0cb", "#ee946c", "#da81c1"]
template = "plotly_white"
font_settings = dict(family="Computer Modern, serif", size=27, color="black")
tick_font = dict(size=23, color="black")
margin_settings = dict(l=10, r=10, t=10, b=10)

# Two-model comparison colors
model_colors = ["#7dbfa7", "#da81c1"]  # DVD: green, Random: pink


def get_regression_value(pred):
    """If model predicts 8, treat as 2 for regression error calculation."""
    if pred == 8:
        return 2
    return pred


def parse_predictions(data):
    """Extract ground truth, DVD, and random predictions from data."""
    results = []

    for idx, item in data.items():
        gt = item["ground_truth_module"]

        # Handle empty predictions
        dvd_pred = item["dvd_module"][0] if item["dvd_module"] else None
        random_pred = item["random_sample_8_module"][0] if item["random_sample_8_module"] else None

        # Skip if any prediction is missing
        if dvd_pred is None or random_pred is None:
            continue

        # Convert to int
        dvd_pred = int(dvd_pred)
        random_pred = int(random_pred)

        # Extract module letter (A, B, C, D, E)
        module = item["module"].split(":")[0].strip()

        results.append({
            "idx": int(idx),
            "module": module,
            "test_type": item["test_type"],
            "ground_truth": gt,
            "dvd_pred": dvd_pred,
            "random_pred": random_pred
        })

    return results


def calculate_metrics(results, model_key):
    """Calculate accuracy, F1, and regression error for a model."""
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r[model_key] for r in results]

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # F1 (weighted for multi-class)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Regression error (MAE with 8->2 conversion)
    reg_errors = []
    for gt, pred in zip(y_true, y_pred):
        gt_reg = get_regression_value(gt)
        pred_reg = get_regression_value(pred)
        reg_errors.append(abs(gt_reg - pred_reg))
    mae = np.mean(reg_errors)

    return {"accuracy": acc, "f1": f1, "mae": mae}


def calculate_metrics_by_module(results, model_key):
    """Calculate metrics grouped by module."""
    module_results = defaultdict(list)

    for r in results:
        module_results[r["module"]].append(r)

    metrics = {}
    for module, mod_results in sorted(module_results.items()):
        metrics[module] = calculate_metrics(mod_results, model_key)

    return metrics


def create_global_comparison_bar(dvd_metrics, random_metrics, metric_name, y_label, filename, y_range=None):
    """Create bar plot comparing DVD vs Random for a single metric."""
    models = ["DVD Agent", "Random Sample"]
    values = [dvd_metrics[metric_name], random_metrics[metric_name]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color=model_colors,
        width=0.6
    ))

    # Add value annotations
    y_span = y_range[1] - y_range[0] if y_range else 0.1
    margin_offset = y_span * 0.08

    for i, (mod, val) in enumerate(zip(models, values)):
        fig.add_annotation(
            x=mod,
            y=val - margin_offset,
            text=f"{val:.3f}",
            showarrow=False,
            font=dict(size=22, color="white", family="Computer Modern, serif"),
            textangle=-90
        )

    layout_kwargs = dict(
        width=500,
        height=500,
        template=template,
        font=font_settings,
        xaxis_title="Model",
        yaxis_title=y_label,
        xaxis=dict(
            title_font=dict(size=27, color="black"),
            tickfont=tick_font,
            tickangle=0
        ),
        yaxis=dict(
            title_font=dict(size=27, color="black"),
            tickfont=tick_font
        ),
        showlegend=False,
        margin=margin_settings
    )

    if y_range:
        layout_kwargs["yaxis"]["range"] = y_range

    fig.update_layout(**layout_kwargs)
    fig.write_image(filename, width=500, height=500, scale=4)
    print(f"Saved: {filename}")


def create_module_comparison_grouped_bar(dvd_by_module, random_by_module, metric_name, y_label, filename, y_range=None):
    """Create grouped bar plot comparing DVD vs Random by module."""
    modules = sorted(dvd_by_module.keys())
    dvd_values = [dvd_by_module[m][metric_name] for m in modules]
    random_values = [random_by_module[m][metric_name] for m in modules]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="DVD Agent",
        x=modules,
        y=dvd_values,
        marker_color=model_colors[0],
        width=0.35,
        offset=-0.18
    ))

    fig.add_trace(go.Bar(
        name="Random Sample",
        x=modules,
        y=random_values,
        marker_color=model_colors[1],
        width=0.35,
        offset=0.18
    ))

    # Add value annotations on bars using text property
    y_span = y_range[1] - y_range[0] if y_range else 0.1
    margin_offset = y_span * 0.12

    # Update traces with text annotations
    fig.update_traces(
        text=[f"{v:.3f}" for v in dvd_values],
        textposition="inside",
        textangle=-90,
        textfont=dict(size=18, color="white", family="Computer Modern, serif"),
        selector=dict(name="DVD Agent")
    )
    fig.update_traces(
        text=[f"{v:.3f}" for v in random_values],
        textposition="inside",
        textangle=-90,
        textfont=dict(size=18, color="white", family="Computer Modern, serif"),
        selector=dict(name="Random Sample")
    )

    layout_kwargs = dict(
        width=700,
        height=500,
        template=template,
        font=font_settings,
        xaxis_title="Module",
        yaxis_title=y_label,
        xaxis=dict(
            title_font=dict(size=27, color="black"),
            tickfont=tick_font,
            tickangle=0
        ),
        yaxis=dict(
            title_font=dict(size=27, color="black"),
            tickfont=tick_font
        ),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            font=dict(size=20)
        ),
        barmode="group",
        bargap=0.15,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    if y_range:
        layout_kwargs["yaxis"]["range"] = y_range

    fig.update_layout(**layout_kwargs)
    fig.write_image(filename, width=700, height=500, scale=4)
    print(f"Saved: {filename}")


def create_question_heatmap(results, metric_type, filename):
    """Create heatmap showing per-question signed errors (pred - gt) for both models."""
    # Sort by index
    results_sorted = sorted(results, key=lambda x: x["idx"])

    # Calculate per-question values
    questions = []
    dvd_values = []
    random_values = []

    for r in results_sorted:
        # Short label
        test_label = r["test_type"].split(".")[0] + "." + r["test_type"].split(".")[1][:2]
        questions.append(test_label)

        gt = r["ground_truth"]
        dvd = r["dvd_pred"]
        random = r["random_pred"]

        if metric_type == "correct":
            # 1 if correct, 0 if wrong
            dvd_val = 1 if dvd == gt else 0
            random_val = 1 if random == gt else 0
        elif metric_type == "error":
            # Signed regression error (pred - gt): positive = overestimate, negative = underestimate
            gt_reg = get_regression_value(gt)
            dvd_val = get_regression_value(dvd) - gt_reg
            random_val = get_regression_value(random) - gt_reg

        dvd_values.append(dvd_val)
        random_values.append(random_val)

    # Create heatmap with 2 rows: DVD, Random (no difference row)
    z_data = [dvd_values, random_values]
    y_labels = ["DVD Agent", "Random Sample"]

    fig = go.Figure()

    # Determine colorscale based on metric type
    if metric_type == "correct":
        # For correctness: green = correct (1), pink = incorrect (0)
        colorscale = [[0, "#da81c1"], [1, "#7dbfa7"]]
        zmin, zmax = 0, 1
        colorbar_title = "Correct"
    else:
        # For signed error: green = underestimate (negative), white = correct, pink = overestimate (positive)
        colorscale = [[0, "#7dbfa7"], [0.5, "white"], [1, "#da81c1"]]
        zmin, zmax = -3, 3
        colorbar_title = "Error"

    fig.add_trace(go.Heatmap(
        z=z_data,
        x=questions,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=[[f"{v}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        showscale=True,
        colorbar=dict(
            title=dict(text=colorbar_title, font=dict(size=18)),
            tickfont=dict(size=14)
        )
    ))

    fig.update_layout(
        width=1400,
        height=280,
        template=template,
        font=dict(family="Computer Modern, serif", size=14, color="black"),
        xaxis=dict(
            title="Question",
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=10, color="black"),
            tickangle=45
        ),
        yaxis=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=14, color="black")
        ),
        margin=dict(l=120, r=80, t=30, b=100)
    )

    fig.write_image(filename, width=1400, height=280, scale=4)
    print(f"Saved: {filename}")


def create_per_question_bar(results, filename):
    """Create bar chart showing correct/incorrect per question for both models."""
    results_sorted = sorted(results, key=lambda x: x["idx"])

    questions = []
    dvd_correct = []
    random_correct = []

    for r in results_sorted:
        test_label = r["test_type"].split(".")[0] + "." + r["test_type"].split(".")[1][:3]
        questions.append(test_label)

        gt = r["ground_truth"]
        dvd_correct.append(1 if r["dvd_pred"] == gt else 0)
        random_correct.append(1 if r["random_pred"] == gt else 0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="DVD Agent",
        x=questions,
        y=dvd_correct,
        marker_color=model_colors[0],
        width=0.35,
        offset=-0.18
    ))

    fig.add_trace(go.Bar(
        name="Random Sample",
        x=questions,
        y=random_correct,
        marker_color=model_colors[1],
        width=0.35,
        offset=0.18
    ))

    fig.update_layout(
        width=1400,
        height=400,
        template=template,
        font=dict(family="Computer Modern, serif", size=14, color="black"),
        xaxis_title="Question",
        yaxis_title="Correct (1) / Incorrect (0)",
        xaxis=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=10, color="black"),
            tickangle=45
        ),
        yaxis=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=14, color="black"),
            range=[0, 1.1]
        ),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            font=dict(size=16)
        ),
        barmode="group",
        margin=dict(l=60, r=20, t=30, b=120)
    )

    fig.write_image(filename, width=1400, height=400, scale=4)
    print(f"Saved: {filename}")


def create_regression_error_comparison(results, filename):
    """Create bar chart showing signed regression error per question (pred - gt)."""
    results_sorted = sorted(results, key=lambda x: x["idx"])

    questions = []
    dvd_errors = []
    random_errors = []

    for r in results_sorted:
        test_label = r["test_type"].split(".")[0] + "." + r["test_type"].split(".")[1][:3]
        questions.append(test_label)

        gt_reg = get_regression_value(r["ground_truth"])
        dvd_reg = get_regression_value(r["dvd_pred"])
        random_reg = get_regression_value(r["random_pred"])

        # Signed error: positive = overestimate, negative = underestimate
        dvd_errors.append(dvd_reg - gt_reg)
        random_errors.append(random_reg - gt_reg)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="DVD Agent",
        x=questions,
        y=dvd_errors,
        marker_color=model_colors[0],
        width=0.35,
        offset=-0.18
    ))

    fig.add_trace(go.Bar(
        name="Random Sample",
        x=questions,
        y=random_errors,
        marker_color=model_colors[1],
        width=0.35,
        offset=0.18
    ))

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        width=1400,
        height=400,
        template=template,
        font=dict(family="Computer Modern, serif", size=14, color="black"),
        xaxis_title="Question",
        yaxis_title="Signed Error (+ over / − under)",
        xaxis=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=10, color="black"),
            tickangle=45
        ),
        yaxis=dict(
            title_font=dict(size=18, color="black"),
            tickfont=dict(size=14, color="black"),
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1
        ),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            font=dict(size=16)
        ),
        barmode="group",
        margin=dict(l=60, r=20, t=30, b=120)
    )

    fig.write_image(filename, width=1400, height=400, scale=4)
    print(f"Saved: {filename}")


def main():
    # Parse data
    results = parse_predictions(data)
    print(f"Loaded {len(results)} questions with predictions")

    # Calculate global metrics
    dvd_metrics = calculate_metrics(results, "dvd_pred")
    random_metrics = calculate_metrics(results, "random_pred")

    print("\n=== Global Metrics ===")
    print(f"DVD Agent:     Acc={dvd_metrics['accuracy']:.3f}, F1={dvd_metrics['f1']:.3f}, MAE={dvd_metrics['mae']:.3f}")
    print(f"Random Sample: Acc={random_metrics['accuracy']:.3f}, F1={random_metrics['f1']:.3f}, MAE={random_metrics['mae']:.3f}")

    # Calculate per-module metrics
    dvd_by_module = calculate_metrics_by_module(results, "dvd_pred")
    random_by_module = calculate_metrics_by_module(results, "random_pred")

    print("\n=== Per-Module Metrics ===")
    for module in sorted(dvd_by_module.keys()):
        dvd_m = dvd_by_module[module]
        rand_m = random_by_module[module]
        print(f"Module {module}:")
        print(f"  DVD:    Acc={dvd_m['accuracy']:.3f}, F1={dvd_m['f1']:.3f}, MAE={dvd_m['mae']:.3f}")
        print(f"  Random: Acc={rand_m['accuracy']:.3f}, F1={rand_m['f1']:.3f}, MAE={rand_m['mae']:.3f}")

    # Generate plots
    print("\n=== Generating Plots ===")

    # 1. Global comparison bar plots (start y-axis at 0.1 to highlight differences)
    create_global_comparison_bar(dvd_metrics, random_metrics, "accuracy", "Accuracy",
                                  "../outputs/plots/global_accuracy.png", y_range=[0, 1.2])
    create_global_comparison_bar(dvd_metrics, random_metrics, "f1", "F1 Score",
                                  "../outputs/plots/global_f1.png", y_range=[0, 1.2])
    create_global_comparison_bar(dvd_metrics, random_metrics, "mae", "MAE (8→2)",
                                  "../outputs/plots/global_mae.png", y_range=[0, 3])

    # 2. Per-module grouped bar plots (start y-axis higher to highlight differences)
    create_module_comparison_grouped_bar(dvd_by_module, random_by_module, "accuracy",
                                          "Accuracy", "../outputs/plots/module_accuracy.png", y_range=[0, 1.2])
    create_module_comparison_grouped_bar(dvd_by_module, random_by_module, "f1",
                                          "F1 Score", "../outputs/plots/module_f1.png", y_range=[0, 1.2])
    create_module_comparison_grouped_bar(dvd_by_module, random_by_module, "mae",
                                          "MAE (8→2)", "../outputs/plots/module_mae.png", y_range=[0, 3])

    # 3. Per-question heatmaps
    create_question_heatmap(results, "correct", "../outputs/plots/heatmap_correctness.png")
    create_question_heatmap(results, "error", "../outputs/plots/heatmap_error.png")

    # 4. Per-question bar charts
    create_per_question_bar(results, "../outputs/plots/question_correctness.png")
    create_regression_error_comparison(results, "../outputs/plots/question_regression_error.png")

    print("\n=== All plots generated successfully! ===")
    print("Files saved in outputs/plots/")


if __name__ == "__main__":
    import os

    result_json = r"C:/Users/haowe/OneDrive/Desktop/MIT/AIProject/DeepVideoDiscovery/results/luke_module_t.json"

    with open(result_json, "r") as f:
        data = json.load(f)

    os.makedirs("../outputs/plots", exist_ok=True)
    main()