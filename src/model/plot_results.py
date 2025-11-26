import matplotlib.pyplot as plt
import pickle

history_path = "cloud_models/juannico007/Nico_cascade_U_net/history.pickle"
with open(history_path, 'rb') as handle:
    history = pickle.load(handle)

print(history)
fig = plt.figure()
plt.plot(history["train_loss"], label = "train loss")
plt.plot(history["val_loss"], label = "val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
output_path = "losses.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
plt.close(fig)

fig = plt.figure()
plt.plot(history["train_acc"], label = "train acc")
plt.plot(history["val_acc"], label = "val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
output_path = "accuracies.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
plt.close(fig)

fig = plt.figure()
plt.plot(history["train_dice"], label = "train dice")
plt.plot(history["val_dice"], label = "val dice")
plt.xlabel("Epoch")
plt.ylabel("DICE score")
plt.legend()
output_path = "dice.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
plt.close(fig)

metric = "sensitivity"
fig = plt.figure()
plt.plot(history[f"train_{metric}"], label = f"train {metric}")
plt.plot(history[f"val_{metric}"], label = f"val {metric}")
plt.xlabel("Epoch")
plt.ylabel(metric)
plt.legend()
output_path = f"{metric}.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
plt.close(fig)
