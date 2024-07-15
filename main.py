import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV File
# Read the breast cancer dataset from a CSV file into a DataFrame
cancer_data = pd.read_csv('breast-cancer.csv')

# Data Cleaning and Preparation
def preprocess_data():
    # Remove rows with empty cells
    cleaned_data = cancer_data.dropna()

    # Remove the 'id' column as it is not needed for the analysis
    feature_data = cleaned_data.drop('id', axis=1)

    # Split the data into training and testing sets (80% training, 20% testing)
    train_data, test_data = train_test_split(feature_data, test_size=0.2, random_state=42)

    # Save the 'diagnosis' column separately for training and testing sets
    global train_labels
    train_labels = train_data['diagnosis']

    global test_labels
    test_labels = test_data['diagnosis']

    # Remove the 'diagnosis' column from the training and testing sets
    train_data = train_data.drop('diagnosis', axis=1)
    test_data = test_data.drop('diagnosis', axis=1)

    return train_data, test_data

# Prepare the training and testing datasets
train_features, test_features = preprocess_data()

# Training the dataset on the Support Vector Machine (RBF)
def train_rbf_svc(train_data, test_data, train_labels, test_labels):
    # Standardize the features
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Select the first two features for training and testing
    X_train = train_data_scaled[:, :2]
    X_test = test_data_scaled[:, :2]

    # Train the SVM model with RBF kernel
    start_time = time.time()
    svm_rbf = SVC(kernel='rbf', random_state=42).fit(X_train, train_labels)
    training_time = time.time() - start_time

    # Predict using the SVM model
    svm_rbf_pred = svm_rbf.predict(X_test)

    # Calculate accuracy, confusion matrix, sensitivity, and specificity
    accuracy = accuracy_score(test_labels, svm_rbf_pred)
    cm = confusion_matrix(test_labels, svm_rbf_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])

    # Print the model performance metrics
    print(f"Time taken: {training_time} seconds")
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n")

    # Visualize the class boundary using 2 features
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the class for each point in the grid
    Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Map classes to numerical values for visualization
    Z[Z == 'M'] = 1
    Z[Z == 'B'] = 0
    Z = np.array(Z, dtype=float)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=[1 if label == 'M' else 0 for label in train_labels], edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title('SVM Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Train the dataset on different SVM kernels and compare performance
def plot_decision_boundary(svm_model, scaler, train_data, train_labels):
    # Standardize the training data
    train_data_scaled = scaler.transform(train_data)

    # Select the first two features for visualization
    X_train = train_data_scaled[:, :2]

    # Create a mesh grid for plotting
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the class for each point in the grid
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = svm_model.predict(xy)
    Z[Z == 'M'] = 1
    Z[Z == 'B'] = 0
    Z = np.array(Z, dtype=float)        
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=[1 if label == 'M' else 0 for label in train_labels], edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title('SVM Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def evaluate_model(svm_model, scaler, test_data, test_labels):
    # Standardize the testing data
    test_data_scaled = scaler.transform(test_data)

    # Use only the first two features for testing
    X_test = test_data_scaled[:, :2]

    # Predict using the SVM model
    svm_pred = svm_model.predict(X_test)

    # Calculate accuracy, confusion matrix, sensitivity, and specificity
    accuracy = accuracy_score(test_labels, svm_pred)
    cm = confusion_matrix(test_labels, svm_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])

    # Print the model performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n")

def train_svm_model(kernel_type, train_data, train_labels):
    # Standardize the training data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # Use only the first two features for training
    X_train = train_data_scaled[:, :2]

    # Train the SVM model with the specified kernel
    svm_model = SVC(kernel=kernel_type, random_state=42)
    svm_model.fit(X_train, train_labels)

    return svm_model, scaler

def train_and_compare_svm_kernels():
    start_time = time.time()
    # Train SVM models with RBF, linear, and polynomial kernels in parallel
    with mp.Pool(processes=3) as pool:
        results = pool.starmap(train_svm_model, [('rbf', train_features, train_labels),
                                                 ('linear', train_features, train_labels),
                                                 ('poly', train_features, train_labels)])
    time_total = time.time() - start_time
    print(f"Training time using MP: {time_total} seconds")

    # Evaluate and visualize each SVM model
    for (svm_model, scaler), kernel in zip(results, ['rbf', 'linear', 'poly']):
        print(f"\nResults for SVM with {kernel} kernel:")
        plot_decision_boundary(svm_model, scaler, train_features, train_labels)
        evaluate_model(svm_model, scaler, test_features, test_labels)

    # Ensemble prediction using majority voting
    predictions = [svm.predict(scaler.transform(test_features)[:, :2]) for svm, scaler in results]
    ensemble_pred = []
    for i in range(len(predictions[0])):
        votes = [pred[i] for pred in predictions]
        ensemble_pred.append(max(set(votes), key=votes.count))

    # Evaluate ensemble
    print("\nEnsemble Results:")
    accuracy = accuracy_score(test_labels, ensemble_pred)
    cm = confusion_matrix(test_labels, ensemble_pred)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])

    # Print ensemble performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Confusion Matrix:\n{cm}")

# Execute the train_and_compare_svm_kernels function if the script is run directly
if __name__ == '__main__':
    train_rbf_svc(train_features, test_features, train_labels, test_labels)
    train_and_compare_svm_kernels()
