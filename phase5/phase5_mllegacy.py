# =============================================================================
# phase5_ml.py — Complete Phase 5 ML Pipeline
# Solar-Wind-Biomass Decision Support System | Odisha | Block Level (314 Samples)
# Includes: Phase 5 Core + All 5 Enhancements
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     GridSearchCV)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── OUTPUT FOLDER ────────────────────────────────────────────────────────────
# Change this path if you want output saved elsewhere
OUTPUT_DIR = r"C:\Users\KIIT0001\OneDrive\Desktop\stuff\miniproject\phase5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# Update this path to wherever your block_features.csv lives
DATA_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\stuff\block_features.csv"

df = pd.read_csv(DATA_PATH)
print(f'\n{"="*60}')
print(f'Blocks loaded: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'{"="*60}\n')

features = [
    'solar_mean', 'wind_mean', 'pop_mean',
    'dist_roads_mean', 'dist_trans_mean',
    'dist_sub_mean', 'constraint_pct'
]
X = df[features].copy()

# =============================================================================
# ENHANCEMENT 1 — Feature Correlation Heatmap (Multicollinearity Check)
# =============================================================================
print('--- Enhancement 1: Feature Correlation Heatmap ---')

plt.figure(figsize=(10, 8))
corr_matrix = df[features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix — Multicollinearity Check', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150)
plt.close()
print('Saved: correlation_heatmap.png')

# Flag highly correlated pairs (above 0.85)
high_corr = [
    (features[i], features[j], corr_matrix.iloc[i, j])
    for i in range(len(features))
    for j in range(i + 1, len(features))
    if abs(corr_matrix.iloc[i, j]) > 0.85
]
if high_corr:
    print('High correlation pairs (>0.85):')
    for f1, f2, val in high_corr:
        print(f'  {f1} vs {f2}: {val:.3f}')
else:
    print('No multicollinearity detected above 0.85 threshold')

# =============================================================================
# STEP 5.1 — K-Means Clustering (Unsupervised ML)
# =============================================================================
print('\n--- Step 5.1: K-Means Clustering ---')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ENHANCEMENT 2 — Elbow Method & Silhouette for Optimal k
print('--- Enhancement 2: Elbow + Silhouette Analysis ---')

inertias = []
silhouette_scores_list = []
k_range = range(2, 11)

for k in k_range:
    km_tmp = KMeans(n_clusters=k, init='k-means++',
                    random_state=42, n_init=10)
    labels_tmp = km_tmp.fit_predict(X_scaled)
    inertias.append(km_tmp.inertia_)
    silhouette_scores_list.append(silhouette_score(X_scaled, labels_tmp))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=4, color='red', linestyle='--', label='k=4 (chosen)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax1.set_title('Elbow Method for Optimal k')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(list(k_range), silhouette_scores_list, 'ro-', linewidth=2, markersize=8)
ax2.axvline(x=4, color='blue', linestyle='--', label='k=4 (chosen)')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score for Optimal k')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Optimal k Selection — Elbow + Silhouette', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/optimal_k_selection.png', dpi=150)
plt.close()

best_k = list(k_range)[silhouette_scores_list.index(max(silhouette_scores_list))]
print(f'Optimal k by Silhouette: {best_k}')
print('Saved: optimal_k_selection.png')

# Run final K-Means with k=4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, df['cluster'])
print(f'Silhouette Score (k=4): {score:.3f}')
df[['block_name', 'cluster']].to_csv(f'{OUTPUT_DIR}/block_clusters.csv', index=False)
print('Saved: block_clusters.csv')

# Cluster distribution plot
plt.figure(figsize=(8, 5))
cluster_counts = df['cluster'].value_counts().sort_index()
plt.bar([f'Cluster {i}' for i in cluster_counts.index],
        cluster_counts.values, color=['steelblue', 'coral', 'green', 'purple'])
plt.title(f'K-Means Cluster Distribution (k=4, Silhouette={score:.3f})')
plt.ylabel('Number of Blocks')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cluster_distribution.png', dpi=150)
plt.close()

# =============================================================================
# STEP 5.2 — Auto Label Generation
# =============================================================================
print('\n--- Step 5.2: Auto Label Generation ---')

max_solar = df['solar_mean'].max()
max_wind  = df['wind_mean'].max()
max_pop   = df['pop_mean'].max()

def assign_label(row):
    norm = {
        'SOLAR':   row['solar_mean'] / max_solar,
        'WIND':    row['wind_mean']  / max_wind,
        'BIOMASS': row['pop_mean']   / max_pop
    }
    best = max(norm, key=norm.get)
    s = sorted(norm.values(), reverse=True)
    return 'HYBRID' if (s[0] - s[1]) < 0.15 else best

df['label'] = df.apply(assign_label, axis=1)
print('Label distribution:')
print(df['label'].value_counts())

# =============================================================================
# STEP 5.3 — Random Forest (with Enhancement 3: GridSearchCV)
# =============================================================================
print('\n--- Step 5.3: Random Forest + Enhancement 3: GridSearchCV ---')

y = df['label']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ENHANCEMENT 3 — GridSearchCV Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 3, 5]
}
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(
    rf_base, param_grid,
    cv=5, scoring='accuracy',
    n_jobs=-1, verbose=0
)
grid_search.fit(X, y)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best CV Accuracy: {grid_search.best_score_:.3f}')

# Use the best model going forward
rf = grid_search.best_estimator_
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f'Tuned RF CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')

rf.fit(X, y)
df['rf_prediction'] = rf.predict(X)
df['rf_confidence'] = rf.predict_proba(X).max(axis=1)
print('\nClassification Report:')
print(classification_report(y, df['rf_prediction']))

joblib.dump(rf, f'{OUTPUT_DIR}/random_forest_model.pkl')
print('Saved: random_forest_model.pkl')

# Confusion matrix heatmap
cm = confusion_matrix(y, df['rf_prediction'],
                      labels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'],
            yticklabels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'])
plt.title('Random Forest — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.close()

# Feature importance chart
imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
plt.title('Random Forest — Feature Importance')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150)
plt.close()
print('Saved: confusion_matrix.png, feature_importance.png')

# =============================================================================
# STEP 5.4 — XGBoost (Algorithm Validation)
# =============================================================================
print('\n--- Step 5.4: XGBoost Comparison ---')

le = LabelEncoder()
y_enc = le.fit_transform(y)
xgb = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='mlogloss', verbosity=0
)
xgb_scores = cross_val_score(xgb, X, y_enc, cv=5, scoring='accuracy')
print(f'XGBoost CV: {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}')
print(f'RF CV:      {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')

model_comparison = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'XGBoost'],
    'CV_Accuracy': [cv_scores.mean(), xgb_scores.mean()],
    'Std_Dev': [cv_scores.std(), xgb_scores.std()]
}).round(4)
model_comparison.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
print(model_comparison.to_string(index=False))
print('Saved: model_comparison.csv')

# Model comparison bar chart
plt.figure(figsize=(7, 5))
bars = plt.bar(model_comparison['Model'], model_comparison['CV_Accuracy'],
               color=['steelblue', 'coral'], alpha=0.85, width=0.4)
plt.errorbar(model_comparison['Model'], model_comparison['CV_Accuracy'],
             yerr=model_comparison['Std_Dev'], fmt='none', color='black',
             capsize=8, linewidth=2)
plt.ylim(0, 1.05)
plt.ylabel('5-Fold CV Accuracy')
plt.title('Model Comparison: Random Forest vs XGBoost')
for bar, val in zip(bars, model_comparison['CV_Accuracy']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=150)
plt.close()
print('Saved: model_comparison.png')

# =============================================================================
# STEP 5.5 — AHP Validation (Expert Weights vs RF Importance)
# =============================================================================
print('\n--- Step 5.5: AHP Validation ---')

ahp = {
    'solar_mean':      0.32,
    'wind_mean':       0.28,
    'pop_mean':        0.15,
    'dist_roads_mean': 0.10,
    'dist_trans_mean': 0.08,
    'dist_sub_mean':   0.04,
    'constraint_pct':  0.03
}
ahp_df = pd.DataFrame({
    'Feature':        features,
    'AHP_Weight':     [ahp[f] for f in features],
    'RF_Importance':  rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

x_pos = range(len(features))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - 0.2 for i in x_pos], ahp_df['AHP_Weight'],  0.4,
       label='AHP Expert Weight', color='steelblue', alpha=0.8)
ax.bar([i + 0.2 for i in x_pos], ahp_df['RF_Importance'], 0.4,
       label='RF Data-Learned Importance', color='coral', alpha=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(ahp_df['Feature'], rotation=30, ha='right')
ax.set_ylabel('Weight / Importance')
ax.set_title('AHP Expert Weights vs Random Forest Feature Importance')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ahp_validation.png', dpi=150)
plt.close()
ahp_df.to_csv(f'{OUTPUT_DIR}/ahp_comparison.csv', index=False)
print('Saved: ahp_validation.png, ahp_comparison.csv')
print(ahp_df.to_string(index=False))

# =============================================================================
# ENHANCEMENT 4 — Feature Ablation Study (Drop-Column Validation)
# =============================================================================
print('\n--- Enhancement 4: Feature Ablation Study ---')

baseline = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()
print(f'Baseline (all features): {baseline:.3f}')

ablation_results = []
for feat in features:
    X_dropped = df[[f for f in features if f != feat]]
    dropped_score = cross_val_score(
        rf, X_dropped, y, cv=cv, scoring='accuracy').mean()
    drop = baseline - dropped_score
    ablation_results.append({
        'Feature': feat,
        'Accuracy_Without': dropped_score,
        'Accuracy_Drop': drop
    })
    print(f'  Remove {feat}: accuracy = {dropped_score:.3f} (drop: {drop:+.3f})')

ablation_df = pd.DataFrame(ablation_results).sort_values(
    'Accuracy_Drop', ascending=False)
ablation_df.to_csv(f'{OUTPUT_DIR}/ablation_study.csv', index=False)

colors = ['red' if x > 0.05 else 'orange' if x > 0.01 else 'green'
          for x in ablation_df['Accuracy_Drop']]
plt.figure(figsize=(10, 5))
plt.bar(ablation_df['Feature'], ablation_df['Accuracy_Drop'], color=colors)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Feature Removed')
plt.ylabel('Accuracy Drop (Baseline − Without Feature)')
plt.title('Feature Ablation Study — Impact of Removing Each Feature\n'
          '(Red >5%, Orange 1-5%, Green <1%)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ablation_study.png', dpi=150)
plt.close()
print('Saved: ablation_study.csv, ablation_study.png')

# =============================================================================
# STEP 5.6 — Confidence Scoring (Handling Uncertainty)
# =============================================================================
print('\n--- Step 5.6: Confidence Scoring ---')

df['final_prediction'] = df.apply(
    lambda row: 'HYBRID' if row['rf_confidence'] < 0.60
    else row['rf_prediction'], axis=1)

high = (df['rf_confidence'] >= 0.80).sum()
med  = ((df['rf_confidence'] >= 0.60) & (df['rf_confidence'] < 0.80)).sum()
low  = (df['rf_confidence'] < 0.60).sum()
print(f'High confidence (>80%): {high} blocks')
print(f'Medium confidence (60-80%): {med} blocks')
print(f'Low confidence (<60%) → override to HYBRID: {low} blocks')
print('\nFinal prediction distribution:')
print(df['final_prediction'].value_counts())

# Confidence distribution plot
plt.figure(figsize=(10, 5))
plt.hist(df['rf_confidence'], bins=20, color='steelblue', edgecolor='white',
         alpha=0.85)
plt.axvline(x=0.60, color='orange', linestyle='--', label='60% threshold (HYBRID)')
plt.axvline(x=0.80, color='green',  linestyle='--', label='80% threshold (High)')
plt.xlabel('Prediction Confidence Score')
plt.ylabel('Number of Blocks')
plt.title('Distribution of RF Prediction Confidence Scores')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confidence_distribution.png', dpi=150)
plt.close()

# Final map-ready CSV
df.to_csv(f'{OUTPUT_DIR}/block_final_predictions.csv', index=False)
print('\nSaved: block_final_predictions.csv (use this for your GIS map)')
print(f'       confidence_distribution.png')

# =============================================================================
# ENHANCEMENT 5 — Block vs District MAUP Comparison
# (Instructions — run this AFTER you have district_features.csv)
# =============================================================================
print('\n--- Enhancement 5: MAUP Block vs District Comparison ---')
print('NOTE: Run this section AFTER you have district_results.csv')
print('      Steps:')
print('      1. Rename block_final_predictions.csv → block_results.csv')
print('      2. Change DATA_PATH to your district_features.csv')
print('      3. Re-run script → rename output to district_results.csv')
print('      4. Uncomment the block below and run once more')

# ---------- UNCOMMENT AFTER YOU HAVE BOTH RESULT FILES ----------
# block_df    = pd.read_csv(f'{OUTPUT_DIR}/block_results.csv')
# district_df = pd.read_csv(f'{OUTPUT_DIR}/district_results.csv')
#
# block_summary = block_df.groupby(
#     ['district_n', 'final_prediction']).size().reset_index()
# block_summary.columns = ['District', 'Energy_Type', 'Block_Count']
#
# block_dominant = (block_df.groupby('district_n')['final_prediction']
#                   .agg(lambda x: x.value_counts().index[0]).reset_index())
# block_dominant.columns = ['District', 'Block_Level_Dominant']
#
# comparison = block_dominant.merge(
#     district_df[['district_n', 'final_prediction']].rename(
#         columns={'district_n': 'District',
#                  'final_prediction': 'District_Level'}),
#     on='District')
#
# comparison['Agreement'] = comparison.apply(
#     lambda r: 'AGREE' if r['Block_Level_Dominant'] == r['District_Level']
#     else 'DISAGREE', axis=1)
#
# print('\nBlock vs District Scale Comparison:')
# print(comparison.to_string(index=False))
# disagreements = (comparison['Agreement'] == 'DISAGREE').sum()
# print(f'\nDisagreements: {disagreements} / {len(comparison)} districts')
# comparison.to_csv(f'{OUTPUT_DIR}/scale_comparison.csv', index=False)
# print('Saved: scale_comparison.csv')
# ----------------------------------------------------------------

# =============================================================================
# SUMMARY
# =============================================================================
print(f'\n{"="*60}')
print('PHASE 5 COMPLETE — ALL OUTPUTS SAVED TO:', OUTPUT_DIR)
print('='*60)
print(f'  RF CV Accuracy:      {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')
print(f'  XGBoost CV Accuracy: {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}')
print(f'  K-Means Silhouette:  {score:.3f}')
print(f'  Best k by silhouette:{best_k}')
print(f'  Best RF params:      {grid_search.best_params_}')
print(f'\n  Final prediction counts:')
for label, count in df["final_prediction"].value_counts().items():
    print(f'    {label:10s}: {count} blocks')
print(f'\n  Output files:')
output_files = [
    'block_clusters.csv', 'block_final_predictions.csv',
    'model_comparison.csv', 'ahp_comparison.csv', 'ablation_study.csv',
    'correlation_heatmap.png', 'optimal_k_selection.png',
    'cluster_distribution.png', 'confusion_matrix.png',
    'feature_importance.png', 'model_comparison.png',
    'ahp_validation.png', 'ablation_study.png',
    'confidence_distribution.png', 'random_forest_model.pkl'
]
for f in output_files:
    print(f'    {f}')
print('='*60)
