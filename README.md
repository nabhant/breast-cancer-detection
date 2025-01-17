# ML Breast Cancer Detection
This project focuses on detecting breast cancer by classifying tumors as either malignant or benign using machine learning techniques. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset. The project includes data cleaning, training various Support Vector Machine (SVM) classifiers, and comparing their performance. The data may be found on [Kaggle.](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

Below you will see the project split into 4 major tasks and what they require:

## Data Reading and Cleaning
- Read the dataset from a CSV file.
- Remove rows with empty cells and unnecessary columns.
- Split the data into training (80%) and testing (20%) sets.

## Data Preparation:
- Standardize the features using StandardScaler.
- Separate features and labels for training and testing.

## SVM Classifier with RBF Kernel:
- Train an SVM classifier using the RBF kernel.
- Evaluate the model's accuracy, sensitivity, and specificity.
- Visualize the decision boundary using the first two features.

## Comparison of Different SVM Kernels: 
- Train SVM classifiers with RBF, linear, and polynomial kernels in parallel.
- Evaluate and visualize each model's performance.
- Use majority voting for ensemble prediction and evaluate the ensemble model's performance.
