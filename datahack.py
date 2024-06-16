import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')


test_features = pd.read_csv('test_set_features.csv')
submission_format = pd.read_csv('submission_format.csv')


X = train_features.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y_xyz = train_labels['xyz_vaccine']
y_seasonal = train_labels['seasonal_vaccine']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])


model = RandomForestClassifier()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


X_train_xyz, X_test_xyz, y_train_xyz, y_test_xyz = train_test_split(X, y_xyz, test_size=0.2, random_state=42)


pipeline.fit(X_train_xyz, y_train_xyz)
predicted_xyz = pipeline.predict_proba(X_test_xyz)[:, 1]
roc_auc_xyz = roc_auc_score(y_test_xyz, predicted_xyz)
print(f'ROC AUC for xyz_vaccine: {roc_auc_xyz}')


X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(X, y_seasonal, test_size=0.2, random_state=42)


pipeline.fit(X_train_seasonal, y_train_seasonal)
predicted_seasonal = pipeline.predict_proba(X_test_seasonal)[:, 1]
roc_auc_seasonal = roc_auc_score(y_test_seasonal, predicted_seasonal)
print(f'ROC AUC for seasonal_vaccine: {roc_auc_seasonal}')


average_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2
print(f'Average ROC AUC: {average_roc_auc}')
