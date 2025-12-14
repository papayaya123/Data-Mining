
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load Data
x_train = pd.read_csv('hw3_lda_knn_face_recog/x_train.csv').iloc[1:,:].astype(float)
y_train = pd.read_csv('hw3_lda_knn_face_recog/y_train.csv').astype(int).to_numpy().ravel()
x_test = pd.read_csv('hw3_lda_knn_face_recog/x_test.csv').iloc[1:,:].astype(float)
y_test = pd.read_csv('hw3_lda_knn_face_recog/y_test.csv').astype(int).to_numpy().ravel()

print(f"Train shapes: X={x_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={x_test.shape}, y={y_test.shape}")

# Strategy 1: PCA -> LDA -> KNN
print("\nTesting PCA -> LDA -> KNN...")
pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("pca", PCA(n_components=0.95)),
    ("lda", LinearDiscriminantAnalysis()),
    ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"))
])

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"PCA+LDA+KNN Accuracy: {acc:.4f}")

# Strategy 2: Direct LDA -> KNN (Baseline)
print("\nTesting Direct LDA -> KNN (Baseline)...")
pipe_base = Pipeline([
    ("scaler", MinMaxScaler()),
    ("lda", LinearDiscriminantAnalysis()),
    ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"))
])
pipe_base.fit(x_train, y_train)
y_pred_base = pipe_base.predict(x_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Direct LDA+KNN Accuracy: {acc_base:.4f}")

# Strategy 3: LDA with Shrinkage
print("\nTesting LDA with Shrinkage ('auto')...")
pipe_shrink = Pipeline([
    ("scaler", MinMaxScaler()),
    ("lda", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"))
])
pipe_shrink.fit(x_train, y_train)
y_pred_shrink = pipe_shrink.predict(x_test)
acc_shrink = accuracy_score(y_test, y_pred_shrink)
print(f"LDA (Shrinkage)+KNN Accuracy: {acc_shrink:.4f}")
