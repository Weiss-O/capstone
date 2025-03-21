import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(filenames):
    metrics = ["TP/FP Ratio", "Precision", "Recall", "F1 Score"]
    data = {}
    
    for file in filenames:
        df = pd.read_csv(file)
        df["Scene_Position"] = df["Scene"] + "_" + df["Position"].astype(str)
        data[file] = df
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for file, df in data.items():
            ax.plot(df["Scene_Position"], df[metric], marker='o', linestyle='-', label=file)
        ax.set_title(metric)
        ax.set_xticklabels(df["Scene_Position"], rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file1> <csv_file2> ...")
        sys.exit(1)
    plot_metrics(sys.argv[1:])

"""
Overall Metrics for detection_summary_22:02:10.csv:
{'Scene_Position': 'Overall', 'TP/FP Ratio': 1.2395833333333333, 'Precision': 0.5534883720930233, 'Recall': 0.5021097046413502, 'F1 Score': 0.5265486725663717}
Overall Metrics for detection_summary_22:51:04.csv:
{'Scene_Position': 'Overall', 'TP/FP Ratio': 1.686046511627907, 'Precision': 0.6277056277056277, 'Recall': 0.6041666666666666, 'F1 Score': 0.6157112526539279}
"""