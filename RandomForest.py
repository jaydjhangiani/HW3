import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV 



# Load the dataset
file_path = './Breast_Cancer_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows to inspect the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 1: Handling Missing Values
# Fill numerical columns with mean and categorical columns with mode
numeric_imputer = SimpleImputer(strategy='mean')
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Impute categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_columns = df.select_dtypes(include=['object']).columns

# Step 2: Replace any missing values ("" or pd.NA) with np.nan for consistency
df[categorical_columns] = df[categorical_columns].replace(["", pd.NA], np.nan)

df[numerical_columns] = numeric_imputer.fit_transform(df[numerical_columns])
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Step 3: Encode Categorical Variables
# Encode the target variable 'Status' and any other categorical variables if necessary
label_encoder = LabelEncoder()
df['Status'] = label_encoder.fit_transform(df['Status'])  # Alive -> 0, Deceased -> 1

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display the cleaned and encoded data
print("\nFirst few rows after cleaning and encoding:")
print(df.head())


for col in numerical_columns:
    lower_limit = np.percentile(df[col], 1)
    upper_limit = np.percentile(df[col], 99)
    df[col] = np.clip(df[col], lower_limit, upper_limit)

# Feature Scaling
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Dimensionality Reduction with PCA (Optional)
pca = PCA(n_components=0.95)  # Retain 95% of variance
principal_components = pca.fit_transform(df[numerical_columns])
print("Explained variance by PCA components:", pca.explained_variance_ratio_)

# Combine PCA components back into DataFrame for further analysis if needed
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
df_pca = pd.DataFrame(data=principal_components, columns=pca_columns)

# Optionally, keep PCA-transformed data only for modeling (comment out the line below if not needed)
df = pd.concat([df, df_pca], axis=1)

# Check for missing values
print("\nMissing values in each column after all processing:")
print(df.isnull().sum())

# Display the final dataset after preprocessing
print("\nFirst few rows of the preprocessed data:")
print(df.head())

# Assuming 'Status' is the target column
X = df.drop(columns=['Status'])
y = df['Status']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy/indicator variables

# Scale the data to ensure non-negative values for Chi-Square
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Feature Selection with SelectKBest
selector = SelectKBest(score_func=chi2, k=5)  # Adjust 'k' as needed
X_selected = selector.fit_transform(X, y)

# Get feature scores for ranking
feature_scores = pd.DataFrame({"Feature": X.columns, "Score": selector.scores_})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)
print("\nFeature ranking based on Chi-Square test:")
print(feature_scores)

# Use the top selected features for Naive Bayes further analysis
selected_features = feature_scores.head(5)['Feature'].values  # Top features
X = X[selected_features]  # Update X to only use selected features

# Data Splitting and Model Evaluation
# Split data (e.g., 70% training, 30% testing)
split_ratio = 0.7
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index].values, X[split_index:].values
y_train, y_test = y[:split_index].values, y[split_index:].values

# Step 1: Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Step 2: Train the model on the training data
rf_model.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# Step 4: Evaluate the model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest model accuracy: {rf_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameters and their ranges for the search
rf_param_grid = {
    'n_estimators': [50, 100, 150],       # Number of trees
    'max_depth': [3, 5, 10, None],        # Maximum depth of trees
    'min_samples_split': [2, 5, 10],      # Minimum samples required to split
}

# Initialize GridSearchCV or RandomizedSearchCV
rf_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# For Random Search, you can use RandomizedSearchCV with 'n_iter' to limit number of combinations
# rf_search = RandomizedSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10)

# Fit the search on training data
rf_search.fit(X_train, y_train)

# Get the best parameters and accuracy score
print("Best Random Forest Hyperparameters:", rf_search.best_params_)
best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy with Best Hyperparameters: {rf_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))