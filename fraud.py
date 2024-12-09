import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
from scipy.stats import uniform, randint
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =====================================
# Load Data
# =====================================
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# =====================================
# Basic Feature Engineering 
# =====================================
train['trans_datetime'] = pd.to_datetime(train['trans_date'] + ' ' + train['trans_time'])
test['trans_datetime'] = pd.to_datetime(test['trans_date'] + ' ' + test['trans_time'])

for df in [train, test]:
    df['trans_hour'] = df['trans_datetime'].dt.hour
    df['trans_dayofweek'] = df['trans_datetime'].dt.dayofweek
    df['trans_day'] = df['trans_datetime'].dt.day
    df['age'] = (df['trans_datetime'] - pd.to_datetime(df['dob'])).dt.days / 365.25

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

train['distance'] = haversine_distance(train['lat'], train['long'], train['merch_lat'], train['merch_long'])
test['distance'] = haversine_distance(test['lat'], test['long'], test['merch_lat'], test['merch_long'])

cat_cols = ['category', 'job', 'state', 'gender']
for col in cat_cols:
    freq = train[col].value_counts()
    train[col + '_freq'] = train[col].map(freq)
    test[col + '_freq'] = test[col].map(freq).fillna(0)

# aggregation by cc_num 
cc_agg = train.groupby('cc_num')['amt'].agg(['mean','std']).reset_index()
cc_agg.columns = ['cc_num','cc_amt_mean','cc_amt_std']
cc_agg['cc_amt_std'] = cc_agg['cc_amt_std'].fillna(0)

train = train.merge(cc_agg, on='cc_num', how='left')
test = test.merge(cc_agg, on='cc_num', how='left').fillna({'cc_amt_std':0})

# Drop unused columns
cols_to_drop = [
    'trans_num','trans_date','trans_time','unix_time','first','last','street','city',
    'dob','merchant','merch_lat','merch_long','lat','long','trans_datetime',
    'category','job','state','gender'
]
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1, errors='ignore')

train = train.fillna(-999)
test = test.fillna(-999)

y = train['is_fraud']
X = train.drop('is_fraud', axis=1)

# =====================================
# Hyperparameter Tuning with CV
# =====================================
lgb_model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)

param_dist = {
    'n_estimators': randint(300, 800),
    'num_leaves': randint(20, 100),
    'learning_rate': uniform(0.01, 0.1),
    'min_child_samples': randint(20, 150),
    'colsample_bytree': uniform(0.6, 0.4),
    'subsample': uniform(0.6, 0.4)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    lgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=skf,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)
best_params = random_search.best_params_
print("Best parameters found:", best_params)

# =====================================
# Cross-Validation for Threshold Selection and Ensemble Seeds
# =====================================
# pick a few different random seeds for final training
ensemble_seeds = [42, 123, 2024]
folds = 5

# Collect fold thresholds and predictions from multiple seeds
all_best_thresholds = []

for seed in ensemble_seeds:
    fold_thresholds = []
    skf_seed = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf_seed.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**best_params, class_weight='balanced', random_state=seed)
        model.fit(X_train_fold, y_train_fold)

        y_val_proba = model.predict_proba(X_val_fold)[:,1]

        # Find best threshold for this fold
        thresholds = np.linspace(0, 1, 101)
        best_thr_fold = 0.5
        best_f1_fold = 0
        for thr in thresholds:
            preds = (y_val_proba >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_val_fold, preds, average='binary')
            if f1 > best_f1_fold:
                best_f1_fold = f1
                best_thr_fold = thr

        fold_thresholds.append(best_thr_fold)

    # Average threshold for this seed across folds
    seed_avg_threshold = np.mean(fold_thresholds)
    all_best_thresholds.append(seed_avg_threshold)

final_threshold = np.mean(all_best_thresholds)
print("Final chosen threshold from CV & seeds:", final_threshold)

# =====================================
# Final Model Training with Ensemble
# =====================================
# Train multiple models with different seeds and average predictions
ensemble_preds = []
final_models = []

for seed in ensemble_seeds:
    final_model = lgb.LGBMClassifier(**best_params, class_weight='balanced', random_state=seed)
    final_model.fit(X, y)
    final_models.append(final_model)

# Predict on test set
test_ids = pd.read_csv("data/test.csv", usecols=['id'])
X_test = test.fillna(-999)

# Average ensemble predictions
y_test_proba_ensemble = np.zeros(X_test.shape[0])
for model in final_models:
    y_test_proba_ensemble += model.predict_proba(X_test)[:,1]

y_test_proba_ensemble /= len(final_models)

y_test_pred = (y_test_proba_ensemble >= final_threshold).astype(int)

submission = pd.DataFrame({
    'id': test_ids['id'],
    'is_fraud': y_test_pred
})
submission.to_csv('my_submission.csv', index=False)

print("Submission file created with ensemble and CV-based threshold.")
