import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


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

# # Outlier Detection and Treatment
# plt.figure(figsize=(15, 8))
# for i, col in enumerate(numerical_columns, 1):
#     plt.subplot(2, 3, i)
#     sns.boxplot(x=df[col])
#     plt.title(f'Box plot of {col}')
# plt.tight_layout()
# plt.show()

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

# Use the top selected features for KNN or further analysis
selected_features = feature_scores.head(5)['Feature'].values  # Top features
X = X[selected_features]  # Update X to only use selected features

# Implementing KNN
# Euclidean distance function
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# KNN function for prediction
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_row in X_test:
        # Calculate distance from test_row to each row in X_train
        distances = [euclidean_distance(test_row, train_row) for train_row in X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return predictions

# Data Splitting and Model Evaluation
# Split data (e.g., 70% training, 30% testing)
split_ratio = 0.7
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index].values, X[split_index:].values
y_train, y_test = y[:split_index].values, y[split_index:].values

# Make predictions with KNN
k = 3  # Choose the value of k
y_pred = knn_predict(X_train, y_train, X_test, k)

# Evaluation (calculate accuracy)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\nKNN model accuracy with k={k}: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
