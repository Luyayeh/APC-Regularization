import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pred_f = open(sys.argv[1])
gold_f = open(sys.argv[2])

pred_f = pred_f.readlines()[10:]

predicted_labels = []
ground_truth_labels = []

area = 'head'

for line_p, line_g in zip(pred_f, gold_f):
    if area == 'head':
        assert line_p == line_g
        area = 'body'
        key = line_p.strip()
    elif area == 'body' and line_p == '.\n':
        area = 'head'
    elif area == 'body':
        labels_p = line_p.split()
        labels_g = line_g.split()

        assert len(labels_p) == len(labels_g)

        predicted_labels.extend(labels_p)
        ground_truth_labels.extend(labels_g)

# Get unique labels
unique_labels = np.unique(ground_truth_labels)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels, labels=unique_labels)

# Set the diagonal values to zero
np.fill_diagonal(conf_matrix, 0)

# Normalize the confusion matrix
row_sums = conf_matrix.sum(axis=1)
conf_matrix_norm = conf_matrix.astype('float') / row_sums[:, np.newaxis]

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6)) #10 8 
im = ax.imshow(conf_matrix_norm, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('Normalized Counts')

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
ax.set_title('Confusion Matrix on SNR10')

# Save the plot as an image file

folder_path = '/home/s2293376/Documents/code/bubo/transfer/results/confusion-matrix'

filename = sys.argv[1].replace('/', '-').replace("exp-", "").replace('.log', '.png') # Remove the extension

destination = os.path.join(folder_path, filename)

plt.savefig(destination, dpi = 200)

# Print a message indicating the saved file name
print("Confusion matrix saved as " + filename)

# Close the plot
plt.close(fig)
