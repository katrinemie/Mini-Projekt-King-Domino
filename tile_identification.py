import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ground truth og forudsagte labels
y_true = ["Field", "Lake", "Forest", "Field", "Mine", "Swamp"]
y_pred = ["Field", "Forest", "Forest", "Field", "Mine", "Field"]

# Definer rækkefølgen af labels
labels = ["Field", "Lake", "Forest", "Grassland", "Swamp", "Mine"]

# Generér confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Udskriv confusion matrix
print("Confusion matrix:")
print(cm)

# Visualisering med heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()