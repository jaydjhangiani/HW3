import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
file_path = './Breast_Cancer_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows to inspect the dataset
print("First few rows of the dataset:")
print(df.head())

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

# Convert all features to integers for C4.5 compatibility
X = X.astype(int)

# Feature Selection with SelectKBest
selector = SelectKBest(score_func=chi2, k=5)  # Adjust 'k' as needed
X_selected = selector.fit_transform(X, y)

# Get feature scores for ranking
feature_scores = pd.DataFrame({"Feature": X.columns, "Score": selector.scores_})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)
print("\nFeature ranking based on Chi-Square test:")
print(feature_scores)

# Use the top selected features for C4.5 Decision Tree further analysis
selected_features = feature_scores.head(5)['Feature'].values  # Top features
X = X[selected_features]  # Update X to only use selected features

# Data Splitting and Model Evaluation
# Split data (e.g., 70% training, 30% testing)
split_ratio = 0.7
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index].values, X[split_index:].values
y_train, y_test = y[:split_index].values, y[split_index:].values



# Define Helper Functions for the C4.5 Decision Tree
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(X, y, feature_index):
    parent_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(y[X[:, feature_index] == values[i]]) for i in range(len(values))]
    )
    return parent_entropy - weighted_entropy

def gain_ratio(X, y, feature_index):
    gain = information_gain(X, y, feature_index)
    split_info = entropy(X[:, feature_index])
    return gain / split_info if split_info > 0 else 0

# C4.5 Decision Tree Class Definition
class DecisionTreeC45:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if len(unique_classes) == 1 or num_samples < self.min_samples_split or depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        # Find the best feature to split on
        best_feature = np.argmax([gain_ratio(X, y, i) for i in range(num_features)])
        tree = {best_feature: {}}

        # Split on the best feature
        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            subtree = self._build_tree(sub_X, sub_y, depth + 1)
            tree[best_feature][value] = subtree

        return tree

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        if not isinstance(tree, dict):
            return tree
        feature_index = list(tree.keys())[0]
        feature_value = inputs[feature_index]
        subtree = tree[feature_index].get(feature_value, None)
        if subtree is None:
            return Counter(y_train).most_common(1)[0][0]
        return self._predict(inputs, subtree)


# Train C4.5 Decision Tree
tree = DecisionTreeC45(min_samples_split=2, max_depth=5)
tree.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = tree.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\nC4.5 Decision Tree accuracy: {accuracy:.2f}")