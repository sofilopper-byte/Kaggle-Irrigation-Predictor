import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

print(f"Train: {len(train)} filas | Test: {len(test)} filas")

cat_cols = ['Soil_Type','Crop_Type','Crop_Growth_Stage','Season',
            'Irrigation_Type','Water_Source','Mulching_Used','Region']

combined = pd.concat([train.drop('Irrigation_Need', axis=1), test], axis=0, ignore_index=True)
le = LabelEncoder()
for col in cat_cols:
    combined[col] = le.fit_transform(combined[col].astype(str))

X_train = combined.iloc[:len(train)].drop('id', axis=1)
X_test  = combined.iloc[len(train):].drop('id', axis=1)
y_train = train['Irrigation_Need']

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
model_test = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=20)
model_test.fit(X_tr, y_tr)
score = balanced_accuracy_score(y_val, model_test.predict(X_val))
print(f"Balanced Accuracy: {score:.4f}")

print("Entrenando modelo final...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=20)
model.fit(X_train, y_train)

submission = pd.DataFrame({'id': test['id'], 'Irrigation_Need': model.predict(X_test)})
submission.to_csv('submission.csv', index=False)
print("Listo!")
print(submission['Irrigation_Need'].value_counts())