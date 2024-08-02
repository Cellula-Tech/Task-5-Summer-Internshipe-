import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import optuna

data = pd.read_csv(r"C:\Users\USER\Downloads\final_internship_data.csv")


null_value = data.isnull().sum()
print("Null values before dropping:")
print(null_value)


data_cleaned = data.dropna()


null_value_after = data_cleaned.isnull().sum()
print("Null values after dropping:")
print(null_value_after)

print("------------------------------------------------------------------------")
print(data_cleaned.dtypes)
print("------------------------------------------------------------------------")


data_cleaned = data_cleaned.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
print("\nData after cleaning:\n", data_cleaned)
print("------------------------------------------------------------------------")


for column in data_cleaned.columns:
    if data_cleaned[column].dtype in ['int64', 'float64']:
        Q1 = data_cleaned[column].quantile(0.25)
        Q3 = data_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mean_value = data_cleaned[column].mean()
        data_cleaned[column] = data_cleaned[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)

print("Data after replacing outliers with mean:")
print(data_cleaned)
print("------------------------------------------------------------------------")


data_cleaned.drop(columns=['User ID', 'User Name', 'Driver Name', 'pickup_datetime', 'key'], inplace=True)


categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_cols, drop_first=True)


X = data_encoded.drop('fare_amount', axis=1)
y = data_encoded['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("------------------------------------------------------------------------")


Featureselection = SelectKBest(score_func=mutual_info_regression, k=20)  # يمكنك تعديل k لاختيار عدد الميزات المطلوب
X_train_selected = Featureselection.fit_transform(X_train, y_train)
X_test_selected = Featureselection.transform(X_test)


selected_indices = Featureselection.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("Selected Features:")
for feature in selected_features:
    print(feature)
print("------------------------------------------------------------------------")

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)  # نطاق أقل للقيم لتسريع التجربة
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)  # نطاق أقل للقيم لتسريع التجربة
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)  # نطاق أقل للقيم لتسريع التجربة
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1) 

print("أفضل المعلمات التي تم العثور عليها: ", study.best_params)


best_params = study.best_params
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)


best_model.fit(X_train_selected, y_train)


y_pred = best_model.predict(X_test_selected)


mae = mean_absolute_error(y_test, y_pred)
print("خطأ متوسط القيمة المطلقة على مجموعة الاختبار: ", mae)
