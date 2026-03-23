import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# =============================
# 1️⃣ SETUP
# =============================
random_state = 42
np.random.seed(random_state)
csv_path = "gestures_control_merged_all.csv"
output_root = "evaluation_results_multi_cv_all"
os.makedirs(output_root, exist_ok=True)

# =============================
# 2️⃣ LOAD & PREPROCESS DATA
# =============================
df = pd.read_csv(csv_path, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Label encoding for compatibility with XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("🧾 Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Feature scaling for MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split for plots and reports
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
)

# =============================
# 3️⃣ DEFINE MODELS
# =============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state),
}

# Try adding XGBoost
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
except ImportError:
    print("⚠️ XGBoost not installed — skipping it (pip install xgboost)")

# =============================
# 4️⃣ TRAIN, EVALUATE & CV
# =============================
results = []
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
f1_macro = make_scorer(f1_score, average='macro')

for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    # ----- Fit & Evaluate on Hold-out Test -----
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)
    acc = accuracy_score(y_test_decoded, y_pred_decoded)
    report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    macro_f1 = report["macro avg"]["f1-score"]

    # ----- Cross-validation -----
    cv_acc = cross_val_score(model, X_scaled, y_encoded, cv=kfold, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_scaled, y_encoded, cv=kfold, scoring=f1_macro)
    cv_acc_mean, cv_acc_std = np.mean(cv_acc), np.std(cv_acc)
    cv_f1_mean, cv_f1_std = np.mean(cv_f1), np.std(cv_f1)

    print(f"✅ {name}: Test Acc={acc:.4f}, CV Acc={cv_acc_mean:.4f}±{cv_acc_std:.4f}")

    # ----- Save Results -----
    model_dir = os.path.join(output_root, name)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, f"{name.lower()}_model.pkl"))
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(model_dir, "classification_report.csv"), index=True)

    df_cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    df_cm.to_csv(os.path.join(model_dir, "confusion_matrix.csv"))

    # Confusion matrix plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Feature importances (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(range(20), importances[idx][:20])
        plt.title(f"Top 20 Feature Importances ({name})")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "feature_importances.png"), dpi=300)
        plt.close()

    # Append results
    results.append({
        "Model": name,
        "Test Accuracy": acc,
        "Test Macro F1": macro_f1,
        "CV Mean Accuracy": cv_acc_mean,
        "CV Std Accuracy": cv_acc_std,
        "CV Mean Macro F1": cv_f1_mean,
        "CV Std Macro F1": cv_f1_std
    })

# =============================
# 5️⃣ SAVE COMPARISON TABLE
# =============================
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_root, "model_comparison_cv.csv"), index=False)
print("\n✅ Cross-validation summary:")
print(df_results)

# =============================
# 6️⃣ PERFORMANCE BAR CHART
# =============================
plt.figure(figsize=(9, 5))
sns.barplot(data=df_results.melt(id_vars="Model",
            value_vars=["CV Mean Accuracy", "CV Mean Macro F1"]),
            x="Model", y="value", hue="variable")
plt.title("Model Cross-Validation Performance (Mean Scores)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(output_root, "cv_model_comparison.png"), dpi=300)
plt.close()

# =============================
# 7️⃣ PCA VISUALIZATION (Best Model)
# =============================
best_model_name = df_results.sort_values("Test Accuracy", ascending=False).iloc[0]["Model"]
best_model_path = os.path.join(output_root, best_model_name, f"{best_model_name.lower()}_model.pkl")
best_model = joblib.load(best_model_path)
print(f"\n🏆 Best model: {best_model_name}")

pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_test)
y_pred_best = best_model.predict(X_test)
y_pred_best_decoded = le.inverse_transform(y_pred_best)

plt.figure(figsize=(8, 6))
for label in np.unique(y_test_decoded):
    idx = y_test_decoded == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"{label}", s=30, alpha=0.7)
plt.legend()
plt.title(f"PCA Projection - Test Samples ({best_model_name})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(output_root, "pca_best_model.png"), dpi=300)
plt.close()

# =============================
# 8️⃣ SAVE ENCODER & SCALER
# =============================
joblib.dump(le, os.path.join(output_root, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(output_root, "feature_scaler.pkl"))

# =============================
# 9️⃣ PRINT SUMMARY FOR THESIS
# =============================
print("\n==============================")
print("📊 FINAL CROSS-VALIDATION SUMMARY")
print("==============================")
for r in results:
    print(f"{r['Model']:<15} | Test Acc: {r['Test Accuracy']:.3f} | "
          f"CV Acc: {r['CV Mean Accuracy']:.3f}±{r['CV Std Accuracy']:.3f}")
print("==============================")
print(f"🏁 Best model: {best_model_name}")
print(f"📁 Results saved in: {output_root}")
print("==============================")
