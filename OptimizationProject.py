import pandas as pd # For data manipulation and analysis

from sklearn.preprocessing import LabelEncoder, StandardScaler # For encoding categorical variables and feature scaling | label encoding converts categorical text data into numerical format. Standard scaling standardizes features by removing the mean and scaling to unit variance.

#load dataset 
df = pd.read_csv('telco_customer_churn.csv') # Load the Telco Customer Churn dataset
print("Dataset loaded successfully.")

print("dataset Info:\n")
print(df.info()) # Display dataset information including data types and non-null counts | helps to understand the structure of the data and identify missing values.
print("\n class distribution:\n")
print(df['Churn'].value_counts()) # Display the distribution of the target variable 'Churn' | helps to understand class imbalance in the dataset.
print("\n sample data:\n")
print(df.head()) # Display the first few rows of the dataset | helps to understand the structure and values of the data.

#handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') # Convert 'TotalCharges' to numeric, setting errors to NaN | ensures that non-numeric entries are handled.
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True) # Fill missing values in 'TotalCharges' with the median | helps to maintain data integrity without introducing bias.

#encode categorical variables
label_encoders = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':  # Exclude 'Churn' from encoding
        df[column] = label_encoders.fit_transform(df[column]) # Apply label encoding to categorical columns | converts categorical text data into numerical format.
        
#encode target variable
df['Churn'] = label_encoders.fit_transform(df['Churn']) # Encode the target variable 'Churn' | converts categorical text data into numerical format.

#scale numerical features
scaler = StandardScaler() # Initialize the StandardScaler | standardizes features by removing the mean and scaling to unit variance.
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] # List of numerical features to be scaled 
df[numerical_features] = scaler.fit_transform(df[numerical_features]) # Apply scaling to numerical features | helps to normalize the data for better model performance.

# feature and target
X = df.drop(columns=['Churn']) # Features (all columns except 'Churn')
y = df['Churn'] # Target variable

from sklearn.model_selection import train_test_split # For splitting the dataset into training and testing sets
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% testing

# train a model
from sklearn.ensemble import RandomForestClassifier # For building the Random Forest model | an ensemble learning method for classification tasks.
from sklearn.metrics import classification_report, confusion_matrix # For evaluating model performance | provides detailed metrics for classification tasks.
from sklearn.base import accuracy_score # For calculating accuracy of the model

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42) # Initialize the Random Forest Classifier
rf_model.fit(X_train, y_train) # Train the model on the training data

# Make predictions on the test set
y_pred = rf_model.predict(X_test) # Predict the target variable for the test set
accuracy_initial = accuracy_score(y_test, y_pred) # Calculate the accuracy of the initial model

print("\nInitial Model Accuracy:\n", accuracy_initial) # Display the accuracy of the initial model
print("\n Classification Report:\n", classification_report(y_test, y_pred)) # Display the classification report for detailed metrics

# Define parameter grid for hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 200], # Number of trees in the forest
    'max_depth': [None, 5, 10, 15], # Maximum depth of the tree
    'min_samples_split': [2, 5, 10,20], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4] # Minimum number of samples required to be at a leaf node
}

#initialize RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV # For hyperparameter tuning using randomized search | helps to find the best hyperparameters for the model.

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42), # The model to be optimized
    param_distributions=param_dist, # The parameter grid
    n_iter=20, # Number of parameter settings that are sampled
    scoring='accuracy', # Evaluation metric
    cv=5, # 5-fold cross-validation
    n_jobs=-1 # Use all available cores
    random_state=42 # Ensure reproducibility
)

#perform Randomized Search
random_search.fit(X_train, y_train) # Fit the RandomizedSearchCV to the training data

#best parameters
best_params = random_search.best_params_ # Get the best hyperparameters found by Randomized Search
print("\nBest Hyperparameters:\n", best_params) # Display the best hyperparameters