# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv('./Newweather.csv')


# Data Cleaning Function
def clean_dataset(df):
    """Clean the dataset by removing invalid entries and NaN values."""
    assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()  # Drop rows with NaN values
    return df


# Apply cleaning
df = clean_dataset(df)

# Visualizing the 'weather' column
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='weather')
plt.title("Weather Category Count")
plt.show()

# Encoding the target variable
weather_mapping = {'drizzle': 0, 'rain': 1, 'sun': 2, 'snow': 3, 'fog': 4}
df['weather'] = df['weather'].map(weather_mapping)

# Replace zero values in specific columns with NaN
columns_to_replace = ['precipitation', 'temp_max', 'temp_min', 'wind']
df[columns_to_replace] = df[columns_to_replace].replace(0, np.NaN)

# Drop rows with NaN after replacements
df = df.dropna()

# Splitting the dataset into features (X) and target (y)
X = df.drop(columns=['weather', 'date'])
y = df['weather']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#292 testing----#1168 training--------


# Function to train and evaluate models
def train_and_evaluate_model(model, model_name):
    """Train a given model and evaluate its performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy on {model_name} training set: {model.score(X_train, y_train):.2f}")
    print(f"Accuracy on {model_name} test set: {model.score(X_test, y_test):.3f}")

    # Save the trained model
    filename = f'{model_name.replace(" ", "_").lower()}_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print(f"{model_name} model saved as {filename}")


# # Train and evaluate MLPClassifier
# mlp_classifier = MLPClassifier(random_state=0)
# train_and_evaluate_model(mlp_classifier, "MLP Classifier")
#
# # Train and evaluate KNeighborsClassifier
# knn_classifier = KNeighborsClassifier()
# train_and_evaluate_model(knn_classifier, "K-Nearest Neighbors Classifier")
#
# # Train and evaluate SVC
# svc_classifier = SVC()
# train_and_evaluate_model(svc_classifier, "Support Vector Classifier")

