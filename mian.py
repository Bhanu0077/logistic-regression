import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)

# 1️⃣ Load Data
df = pd.read_csv("C:\\CODE_DEX\\AI\\logistic-regression\\data.csv")

print("✅ Data Loaded Successfully!")
print("📏 Dataset Shape:", df.shape)
print(df.head(), "\n")

# 2️⃣ Handle Missing Values
df = df.dropna()   # or use df.fillna(df.mean())

# 3️⃣ Separate Features and Target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# 4️⃣ Normalize Features (helps model convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Split into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7️⃣ Make Predictions
y_pred = model.predict(X_test)

# 8️⃣ Evaluate Model
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# 9️⃣ ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Heart Disease Prediction")
plt.legend()
plt.show()

# 🔮 Predict New Example (optional)
sample = [[1, 45, 2, 1, 20, 0, 0, 1, 0, 250, 130, 85, 25.5, 78, 90]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("🩺 Predicted TenYearCHD:", "Yes (1)" if prediction[0] == 1 else "No (0)")

# 10️⃣ Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
