import numpy as np
import os

# Create some sample data arrays (replace these with your actual evaluation arrays)
acc = np.array([0.50, 0.60, 0.65, 0.70])
val_acc = np.array([0.48, 0.58, 0.63, 0.68])
loss = np.array([1.2, 0.9, 0.7, 0.5])
val_loss = np.array([1.3, 1.0, 0.8, 0.6])
y_true = np.array([0, 1, 2, 2, 0, 1])
y_pred = np.array([0, 2, 2, 2, 0, 0])

# Define where to save the file (here, in an 'evaluation' folder at the repo's root)
evaluation_folder = os.path.join(os.getcwd(), "evaluation")
os.makedirs(evaluation_folder, exist_ok=True)

# Save the arrays into evaluation_data.npz
np.savez(os.path.join(evaluation_folder, "evaluation_data.npz"), 
         acc=acc, val_acc=val_acc, loss=loss, val_loss=val_loss, 
         y_true=y_true, y_pred=y_pred)

print("evaluation_data.npz has been created in the 'evaluation' folder.")
