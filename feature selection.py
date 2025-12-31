import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

df = df.drop(columns=["filename"])

df['label'] = df['label'].map({'benign': 0, 'malignant': 1})

corr_matrix = df.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.85)
]
print("Dropping highly correlated features:", to_drop)

df_reduced = df.drop(columns=to_drop)

df_reduced.to_csv("selected_features.csv", index=False)
print("Selected features saved to selected_features.csv")

plt.figure(figsize=(12, 8))
sns.heatmap(df_reduced.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap of Selected Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Heatmap saved as correlation_heatmap.png")
