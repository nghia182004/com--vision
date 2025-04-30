from sklearnex import patch_sklearn 
patch_sklearn()
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the pre-trained model
m = joblib.load('/content/com--vision/bovw_svm_model.pkl')

# Load the training and testing data
with open('/content/com--vision/bovw_train_histograms.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('/content/com--vision/bovw_test_histograms.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('/content/com--vision/test_labels', 'rb') as f:
    y_test = pickle.load(f)

# Scale the data
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)

# Make predictions
print("Predicting...")
y_pred = m['svm'].predict(m['scaler'].transform(X_test_s))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('svm_confusion_matrix.png')