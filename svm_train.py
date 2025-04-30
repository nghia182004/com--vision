import pickle
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1) Load BoVW histograms and labels
with open('/content/com--vision/bovw_train_histograms.pkl', 'rb') as f:
    X_train = pickle.load(f)    # shape: (n_train, K)
with open('/content/com--vision/train_labels', 'rb') as f:
    y_train = pickle.load(f)    # shape: (n_train,)

with open('/content/com--vision/bovw_val_histograms.pkl', 'rb') as f:
    X_val = pickle.load(f)
with open('/content/com--vision/val_labels', 'rb') as f:
    y_val = pickle.load(f)

# 2) Feature scaling
#    SVMs often benefit from zero-mean, unit-variance inputs
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

# 3) Instantiate and train the SVM
#    You can try LinearSVC for speed or SVC(kernel='rbf') for non-linear decision boundaries.
svm = SVC(kernel='sigmoid', C=1.0, gamma='scale', max_iter=15000, random_state=42, verbose=True, class_weight='balanced')
svm.fit(X_train_s, y_train)

# 4) Validate: check accuracy on the val set
y_val_pred = svm.predict(X_val_s)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

joblib.dump({'scaler': scaler, 'svm': svm}, '/content/com--vision/bovw_svm_model.pkl')

