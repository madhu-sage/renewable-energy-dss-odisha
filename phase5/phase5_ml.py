import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configuration
OUTPUT_DIR = r"C:\Users\KIIT0001\OneDrive\Desktop\stuff\miniproject\new5op"
DATA_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\stuff\miniproject\phase4_block\block_features.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f'\n{"="*60}\nDataset: {len(df)} blocks loaded\n{"="*60}\n')

features = ['solar_mean', 'wind_mean', 'pop_mean', 'dist_roads_mean', 
            'dist_trans_mean', 'dist_sub_mean', 'constraint_pct']
X = df[features].copy()

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 1: Correlation Heatmap (Multicollinearity Check)
# ═══════════════════════════════════════════════════════════════════════════
print('Enhancement 1: Correlation Analysis')
corr_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix — Multicollinearity Check')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150)
plt.close()

high_corr = [(features[i], features[j], corr_matrix.iloc[i,j])
             for i in range(len(features)) for j in range(i+1, len(features))
             if abs(corr_matrix.iloc[i,j]) > 0.85]
print(f"✓ Multicollinearity: {'None detected' if not high_corr else f'{len(high_corr)} pairs >0.85'}\n")

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 2: Optimal k Selection (Elbow + Silhouette)
# ═══════════════════════════════════════════════════════════════════════════
print('Enhancement 2: Optimal k Selection')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias, silhouette_scores, k_range = [], [], range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(k_range), inertias, 'bo-', lw=2, ms=8)
ax1.axvline(x=4, color='red', ls='--', label='k=4')
ax1.set(xlabel='Number of Clusters (k)', ylabel='Inertia', title='Elbow Method')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(list(k_range), silhouette_scores, 'ro-', lw=2, ms=8)
ax2.axvline(x=4, color='blue', ls='--', label='k=4')
ax2.set(xlabel='Number of Clusters (k)', ylabel='Silhouette Score', title='Silhouette Analysis')
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/optimal_k_selection.png', dpi=150)
plt.close()
print(f"✓ Optimal k: {k_range[silhouette_scores.index(max(silhouette_scores))]} | k=4 score: {silhouette_scores[2]:.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: K-Means Clustering
# ═══════════════════════════════════════════════════════════════════════════
print('Step 1: K-Means Clustering')
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"✓ 4 clusters created | Silhouette: {silhouette_score(X_scaled, df['cluster']):.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Auto Label Generation  
# ═══════════════════════════════════════════════════════════════════════════
print('Step 2: Label Generation')
max_vals = {'solar': df['solar_mean'].max(), 'wind': df['wind_mean'].max(), 'pop': df['pop_mean'].max()}

def assign_label(row):
    norm = {'SOLAR': row['solar_mean']/max_vals['solar'], 
            'WIND': row['wind_mean']/max_vals['wind'],
            'BIOMASS': row['pop_mean']/max_vals['pop']}
    best = max(norm, key=norm.get)
    scores = sorted(norm.values(), reverse=True)
    return 'HYBRID' if (scores[0] - scores[1]) < 0.15 else best

df['label'] = df.apply(assign_label, axis=1)
print(f"✓ Labels: {dict(df['label'].value_counts())}\n")

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 3: GridSearchCV Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
print('Enhancement 3: GridSearchCV Tuning')
y = df['label']
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 3, 5]}
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), 
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X, y)
rf = grid_search.best_estimator_
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"✓ Best params: {grid_search.best_params_} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Random Forest Training
# ═══════════════════════════════════════════════════════════════════════════
print('Step 3: Random Forest Training')
rf.fit(X, y)
df['rf_prediction'] = rf.predict(X)
df['rf_confidence'] = rf.predict_proba(X).max(axis=1)
joblib.dump(rf, f'{OUTPUT_DIR}/model_simple.pkl')

# Confusion Matrix
cm = confusion_matrix(y, df['rf_prediction'], labels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'],
            yticklabels=['SOLAR', 'WIND', 'BIOMASS', 'HYBRID'])
plt.title('Random Forest — Confusion Matrix')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_simple.png', dpi=150)
plt.close()
print(f"✓ Model trained | Accuracy: {(y == df['rf_prediction']).mean():.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: XGBoost Validation
# ═══════════════════════════════════════════════════════════════════════════
print('Step 4: XGBoost Validation')
le = LabelEncoder()
y_enc = le.fit_transform(y)
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, 
                    eval_metric='mlogloss', verbosity=0)
xgb_scores = cross_val_score(xgb, X, y_enc, cv=5, scoring='accuracy')
comparison = pd.DataFrame({'Model': ['Random Forest', 'XGBoost'],
                           'CV_Accuracy': [cv_scores.mean(), xgb_scores.mean()],
                           'Std_Dev': [cv_scores.std(), xgb_scores.std()]})
comparison.to_csv(f'{OUTPUT_DIR}/model_comparison_simple.csv', index=False)
print(f"✓ RF: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | XGB: {xgb_scores.mean():.3f}±{xgb_scores.std():.3f}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: AHP Validation
# ═══════════════════════════════════════════════════════════════════════════
print('Step 5: AHP Validation')
ahp_weights = {'solar_mean': 0.32, 'wind_mean': 0.28, 'pop_mean': 0.15,
               'dist_roads_mean': 0.10, 'dist_trans_mean': 0.08,
               'dist_sub_mean': 0.04, 'constraint_pct': 0.03}

ahp_comp = pd.DataFrame({'Feature': features, 'AHP_Expert': [ahp_weights[f] for f in features],
                         'RF_Learned': rf.feature_importances_}).sort_values('RF_Learned', ascending=False)
ahp_comp.to_csv(f'{OUTPUT_DIR}/ahp_comparison_simple.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ahp_s = ahp_comp.sort_values('AHP_Expert', ascending=True)
ax1.barh(ahp_s['Feature'], ahp_s['AHP_Expert'], color='steelblue')
ax1.set(xlabel='Weight', title='AHP Expert Weights')
rf_s = ahp_comp.sort_values('RF_Learned', ascending=True)
ax2.barh(rf_s['Feature'], rf_s['RF_Learned'], color='coral')
ax2.set(xlabel='Importance', title='Random Forest Learned Importance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_simple.png', dpi=150)
plt.close()
print(f"✓ Top feature - AHP: {ahp_comp.iloc[0]['Feature']}, RF: {ahp_comp.iloc[0]['Feature']}\n")

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 4: Feature Ablation Study
# ═══════════════════════════════════════════════════════════════════════════
print('Enhancement 4: Feature Ablation Study')
baseline = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()
ablation_results = []
for feat in features:
    X_drop = df[[f for f in features if f != feat]]
    score = cross_val_score(rf, X_drop, y, cv=5, scoring='accuracy').mean()
    ablation_results.append({'Feature': feat, 'Accuracy_Without': score, 'Accuracy_Drop': baseline - score})

ablation_df = pd.DataFrame(ablation_results).sort_values('Accuracy_Drop', ascending=False)
ablation_df.to_csv(f'{OUTPUT_DIR}/ablation_study.csv', index=False)

plt.figure(figsize=(10, 5))
colors = ['red' if x > 0.05 else 'orange' if x > 0.01 else 'green' for x in ablation_df['Accuracy_Drop']]
plt.bar(ablation_df['Feature'], ablation_df['Accuracy_Drop'], color=colors)
plt.axhline(y=0, color='black', ls='-', lw=0.5)
plt.xlabel('Feature Removed'); plt.ylabel('Accuracy Drop')
plt.title('Feature Ablation Study — Impact of Removing Each Feature\n(Red >5%, Orange 1-5%, Green <1%)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ablation_study.png', dpi=150)
plt.close()
print(f"✓ Most critical: {ablation_df.iloc[0]['Feature']} (drop: {ablation_df.iloc[0]['Accuracy_Drop']:.3f})\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Confidence Scoring & Final Predictions
# ═══════════════════════════════════════════════════════════════════════════
print('Step 6: Confidence Scoring')
df['final_prediction'] = df.apply(lambda r: 'HYBRID' if r['rf_confidence'] < 0.60 else r['rf_prediction'], axis=1)
high = (df['rf_confidence'] >= 0.80).sum()
med = ((df['rf_confidence'] >= 0.60) & (df['rf_confidence'] < 0.80)).sum()
low = (df['rf_confidence'] < 0.60).sum()
print(f"✓ High(≥80%): {high} | Medium(60-80%): {med} | Low(<60%): {low} → HYBRID")

plt.figure(figsize=(9, 5))
plt.hist(df['rf_confidence'], bins=25, color='steelblue', edgecolor='white', alpha=0.85)
plt.axvline(x=0.60, color='orange', ls='--', lw=2, label='60%')
plt.axvline(x=0.80, color='green', ls='--', lw=2, label='80%')
plt.xlabel('Confidence Score'); plt.ylabel('Number of Blocks')
plt.title('Confidence Score Distribution'); plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confidence_distribution_simple.png', dpi=150)
plt.close()

output_cols = ['block_name', 'cluster', 'label', 'rf_prediction', 'rf_confidence', 'final_prediction'] + features
df[output_cols].to_csv(f'{OUTPUT_DIR}/final_predictions_simple.csv', index=False)

print(f"\n{'='*60}\n✓ PIPELINE COMPLETE\n{'='*60}")
print(f"Outputs: {OUTPUT_DIR}/\n  • All CSV and PNG files generated\n")

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCEMENT 5: Block vs District Scale Comparison (MAUP)
# ═══════════════════════════════════════════════════════════════════════════
DISTRICT_DATA = r"C:\Users\KIIT0001\OneDrive\Desktop\stuff\miniproject\Phase4ouput files\district_features.csv"
DISTRICT_OUT = f'{OUTPUT_DIR}/final_predictions_district.csv'
BLOCK_OUT = f'{OUTPUT_DIR}/final_predictions_simple.csv'

if os.path.exists(DISTRICT_DATA) and os.path.exists(DISTRICT_OUT):
    print('Enhancement 5: MAUP Analysis')
    try:
        block_df = pd.read_csv(BLOCK_OUT)
        district_df = pd.read_csv(DISTRICT_OUT)
        
        if 'district_n' not in block_df.columns and 'district_name' in block_df.columns:
            block_df['district_n'] = block_df['district_name']
        
        block_dom = block_df.groupby('district_n')['final_prediction'].agg(lambda x: x.value_counts().index[0]).reset_index()
        block_dom.columns = ['District', 'Block_Level']
        
        dist_col = 'district_n' if 'district_n' in district_df.columns else 'block_name'
        comparison = block_dom.merge(district_df[[dist_col, 'final_prediction']].rename(
            columns={dist_col: 'District', 'final_prediction': 'District_Level'}), on='District', how='inner')
        
        comparison['Agreement'] = comparison.apply(lambda r: 'AGREE' if r['Block_Level'] == r['District_Level'] else 'DISAGREE', axis=1)
        disagree = (comparison['Agreement'] == 'DISAGREE').sum()
        
        comparison.to_csv(f'{OUTPUT_DIR}/scale_comparison.csv', index=False)
        plt.figure(figsize=(10, 6))
        plt.bar(['Agree', 'Disagree'], [(comparison['Agreement']=='AGREE').sum(), disagree], 
                color=['green', 'orange'], alpha=0.7, edgecolor='black')
        plt.ylabel('Number of Districts')
        plt.title(f'Block vs District Scale Agreement ({disagree}/{len(comparison)} MAUP effects)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/maup_analysis.png', dpi=150)
        plt.close()
        print(f"✓ MAUP: {disagree}/{len(comparison)} districts disagree\n")
    except Exception as e:
        print(f"⚠ MAUP analysis failed: {e}\n")

elif os.path.exists(DISTRICT_DATA):
    print('\nEnhancement 5: MAUP Analysis')
    print(f"ℹ To enable: Change DATA_PATH to district_features.csv and re-run\n")

