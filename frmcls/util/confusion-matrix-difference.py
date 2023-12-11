import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pred_f_baseline = open(sys.argv[1])
pred_f_variation = open(sys.argv[2])
gold_f = open(sys.argv[3])

pred_f_baseline = pred_f_baseline.readlines()[10:]
pred_f_variation = pred_f_variation.readlines()[10:]

predicted_labels_baseline = []
predicted_labels_variation = []
ground_truth_labels = []

area = 'head'

for line_p_baseline, line_p_variation, line_g in zip(pred_f_baseline, pred_f_variation, gold_f):
    if area == 'head':
        assert line_p_baseline == line_p_variation == line_g
        area = 'body'
        key = line_p_baseline.strip()
    elif area == 'body' and line_p_baseline == '.\n':
        area = 'head'
    elif area == 'body':
        labels_p_baseline = line_p_baseline.split()
        labels_p_variation = line_p_variation.split()
        labels_g = line_g.split()

        assert len(labels_p_baseline) == len(labels_p_variation) == len(labels_g)

        predicted_labels_baseline.extend(labels_p_baseline)
        predicted_labels_variation.extend(labels_p_variation)
        ground_truth_labels.extend(labels_g)

# Get unique labels
unique_labels = np.unique(ground_truth_labels)

# Calculate the confusion matrices
conf_matrix_baseline = confusion_matrix(ground_truth_labels, predicted_labels_baseline, labels=unique_labels)
conf_matrix_variation = confusion_matrix(ground_truth_labels, predicted_labels_variation, labels=unique_labels)

# Calculate the variation matrix (Modified - Baseline)
conf_matrix_diffrence = conf_matrix_variation - conf_matrix_baseline

# Set the diagonal values to zero
np.fill_diagonal(conf_matrix_diffrence, 0)

# Normalize the variation matrix
row_sums = np.abs(conf_matrix_diffrence).sum(axis=1)
conf_matrix_norm = conf_matrix_diffrence.astype('float') / row_sums[:, np.newaxis]

# Plot the variation matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=1)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('Variation')

# Set tick labels
ax.set_xticks(np.arange(len(unique_labels)))
ax.set_yticks(np.arange(len(unique_labels)))
ax.set_xticklabels(unique_labels)
ax.set_yticklabels(unique_labels)

# Rotate x-axis tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

# Set axis labels
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

# Hide grid lines
ax.grid(False)

# Set title
ax.set_title('Differentiated Confusion Matrix (SNR10 - Baseline)')

# Save the plot as an image file
folder_path = '/home/s2293376/Documents/code/bubo/transfer/results/confusion-matrix'
filename = sys.argv[2].replace('/', '-').replace("exp-", "").replace('.log', '.png')  # Remove the extension
filename = 'difference-' + filename
destination = os.path.join(folder_path, filename)
plt.savefig(destination, dpi = 200)

# Print a message indicating the saved file name
print("Difference matrix saved as " + filename)

# Close the plot
plt.close(fig)
