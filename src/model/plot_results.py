import matplotlib.pyplot as plt
import pickle
import os

output_folder = "cloud_models/MihaiB-dev/suppression-loss_002/"
history_path = output_folder + "history.pickle"
with open(history_path, 'rb') as handle:
    history = pickle.load(handle)
os.mkdir(output_folder+"visualizations")

def plot_metric(history, metric, output_folder):
    fig = plt.figure()
    plt.plot(history[f"train_{metric}"], label = f"train {metric}")
    plt.plot(history[f"val_{metric}"], label = f"val {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    output_path = output_folder + f"visualizations/{metric}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


for key in history.keys():
    if key.startswith("train"):
        plot_metric(history, key[6:], output_folder)