import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_loss(training_stats):
    training_losses = [t['Training Loss'] for t in training_stats]
    valid_losses = [t['Valid. Loss'] for t in training_stats]

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(training_losses, 'b-o', label="Training")
    plt.plot(valid_losses, 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

def plot_confusion_matrix(preds, labels):
    # Calculate the confusion matrix
    xtick_labels = ['race', 'gender', 'culture', 'other']
    ytick_labels = xtick_labels
    cm = confusion_matrix(labels, preds, normalize='true')
    # Plot the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.3f', xticklabels=xtick_labels, yticklabels=ytick_labels)
    plt.title('Normalized confusion matrix', weight='roman')
    plt.ylabel('Actual label', weight='roman')
    plt.xlabel('Predicted label', weight='roman')