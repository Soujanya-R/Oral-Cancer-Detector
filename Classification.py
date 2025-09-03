import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Load data
df = pd.read_csv("selected_features.csv")

# Features & Labels
X = df.drop(columns=['label'])
y = df['label']

# Split into training/testing
# Split into training/testing first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE on training set only
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========== Model 1: SVM ==========
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\n🔍 SVM Results:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ========== Model 2: KNN ==========
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("\n🔍 KNN Results:")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# ========== Model 3: Naive Bayes ==========
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("\n🔍 Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# ========== Model 4: ANN ==========
ann = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)
print("\n🔍 ANN Results:")
print("Accuracy:", accuracy_score(y_test, ann_pred))
print(classification_report(y_test, ann_pred))
