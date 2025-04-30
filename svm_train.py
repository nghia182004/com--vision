import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1) Load BoVW histograms and labels
with open('/content/com--vision/bovw_train_histograms.pkl', 'rb') as f:
    X_train = pickle.load(f)    # shape: (n_train, K)
with open('/content/com--vision/train_labels.pkl', 'rb') as f:
    y_train = pickle.load(f)    # shape: (n_train,)

with open('/content/com--vision/bovw_val_histograms.pkl', 'rb') as f:
    X_val = pickle.load(f)
with open('/content/com--vision/val_labels.pkl', 'rb') as f:
    y_val = pickle.load(f)

# 2) Feature scaling
#    SVMs often benefit from zero-mean, unit-variance inputs
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

# 3) Instantiate and train the SVM
#    You can try LinearSVC for speed or SVC(kernel='rbf') for non-linear decision boundaries.
svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
svm.fit(X_train_s, y_train)

# 4) Validate: check accuracy on the val set
y_val_pred = svm.predict(X_val_s)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

joblib.dump({'scaler': scaler, 'svm': svm}, 'bovw_svm_model.pkl')

