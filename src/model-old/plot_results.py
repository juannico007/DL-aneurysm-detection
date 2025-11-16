import matplotlib.pyplot as plt
import pickle

history_path = "../history.pickle"
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