# import pandas as pd
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset
# data = pd.read_csv('dataset_heart.csv')

# # Rename columns to standardize names
# data.columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'resting electrocardiographic results', 'max heart rate', 'exercise induced angina', 'oldpeak', 'ST segment', 'major vessels', 'thal', 'heart disease']

# # DATA PREPROCESSING - Checking for duplicate values
# data = data.drop_duplicates()

# # Separating categorical and numerical values
# cate_val = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results', 'exercise induced angina', 'ST segment', 'thal']
# cont_val = ['age', 'resting blood pressure', 'serum cholestoral', 'max heart rate', 'oldpeak', 'major vessels']

# # Apply one-hot encoding to categorical variables
# data = pd.get_dummies(data, columns=cate_val, drop_first=True)

# # Feature scaling
# scaler = StandardScaler()
# data[cont_val] = scaler.fit_transform(data[cont_val])

# # Splitting the dataset into features (X) and target (Y)
# X = data.drop('heart disease', axis=1)
# Y = data['heart disease']

# # Save feature columns
# feature_columns = X.columns.tolist()
# pickle.dump(feature_columns, open('feature_columns.pkl', 'wb'))

# # Splitting the dataset into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

# # Creating and training the Decision Tree Classifier model
# tree_model = DecisionTreeClassifier(random_state=30)
# tree_model.fit(X_train, Y_train)

# # Making predictions on the test set
# Y_pred = tree_model.predict(X_test)

# # Calculating the accuracy score
# accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)

# # Creating a pickle file for the classifier and scaler
# pickle.dump(tree_model, open('heart-disease-prediction-tree-model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

#ACTIVE CODE-------------(87 % ACCURACY)-------------------------------------------------------------------------

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset_heart.csv')

# Rename columns to standardize names
data.columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'resting electrocardiographic results', 'max heart rate', 'exercise induced angina', 'oldpeak', 'ST segment', 'major vessels', 'thal', 'heart disease']

# DATA PREPROCESSING - Checking for duplicate values
data = data.drop_duplicates()

# Separating categorical and numerical values
cate_val = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results', 'exercise induced angina', 'ST segment', 'thal']
cont_val = ['age', 'resting blood pressure', 'serum cholestoral', 'max heart rate', 'oldpeak', 'major vessels']

# Apply one-hot encoding to categorical variables
data = pd.get_dummies(data, columns=cate_val, drop_first=True)

# Feature scaling
scaler = StandardScaler()
data[cont_val] = scaler.fit_transform(data[cont_val])

# Splitting the dataset into features (X) and target (Y)
X = data.drop('heart disease', axis=1)
Y = data['heart disease']

# Save feature columns
feature_columns = X.columns.tolist()
pickle.dump(feature_columns, open('feature_columns.pkl', 'wb'))

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

# Creating and training the Logistic Regression model
log = LogisticRegression()
log.fit(X_train, Y_train)

# Making predictions on the test set
Y_pred = log.predict(X_test)

# Calculating the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Creating a pickle file for the classifier and scaler
pickle.dump(log, open('heart-disease-prediction-logreg-model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
