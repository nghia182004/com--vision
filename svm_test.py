# FOR TESTING ONLY
%matplotlib inline
from sklearnex import patch_sklearn 
patch_sklearn()
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the pre-trained model
m = joblib.load('/content/com--vision/bovw_svm_model.pkl')

with open('/content/com--vision/bovw_test_histograms.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('/content/com--vision/test_labels', 'rb') as f:
    y_test = pickle.load(f)


# Make predictions
print("Predicting...")
y_pred = m['svm'].predict(m['scaler'].transform(X_test))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

class_names = ['mitsubishi_lancer', 'honda_odyssey', 'nissan_maxima', 'ford_explorer', 'honda_civic', 'nissan_altima', 'mitsubishi_outlander', 'ford_escape']

# Plot the confusion matrix
plt.figure(figsize=(8, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('/content/com--vision/svm_confusion_matrix.png')