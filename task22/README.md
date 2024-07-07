# Task 22: MNIST Image Processing

## Description
This task involves processing the MNIST dataset using Python 
libraries. The goal is to apply various image processing 
techniques and build a machine learning model using a Random 
Forest Classifier to recognise handwritten digits. Understanding 
image processing and machine learning is crucial for 
applications in computer vision, OCR, and more.

## Table of Contents
- [Description](#description)
- [Steps](#steps)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Steps
1. **Create a Copy of the MNIST.ipynb File:**
   - Create a copy of the `MNIST.ipynb` file and rename it 
`mnist_task.ipynb`.

2. **Load the MNIST Dataset:**
   - Use a library such as scikit-learn to access the dataset.
     ```python
     from sklearn.datasets import load_digits
     digits = load_digits()
     ```

3. **Split the Data:**
   - Split the data into training and test sets.
   - **Comment:** The purpose of splitting the data into 
training and test sets is to evaluate the performance of the 
model on unseen data, ensuring it generalises well.

4. **Create a Classification Model:**
   - Use the Random Forest Classifier built into scikit-learn to 
create a classification model.
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier()
     ```

5. **Tune a Parameter:**
   - Pick one parameter to tune (e.g., `n_estimators`, 
`max_depth`) and explain why you chose this parameter.
   - **Comment:** I chose to tune `n_estimators` because it 
controls the number of trees in the forest, which can 
significantly impact the model's performance and computational 
efficiency.

6. **Select a Value for the Parameter:**
   - Select a value for the parameter to use during testing on 
the test data and provide a rationale for your choice.
   - **Comment:** I selected `n_estimators=100` because it 
offers a good balance between performance and computational 
cost.

7. **Print the Confusion Matrix:**
   - Print the confusion matrix for your Random Forest model on 
the test set.
     ```python
     from sklearn.metrics import confusion_matrix
     y_pred = model.predict(X_test)
     print(confusion_matrix(y_test, y_pred))
     ```

8. **Report Class Performance:**
   - Report which classes the model struggles with the most.

9. **Report Performance Metrics:**
   - Report the accuracy, precision, recall, and F1-score.
   - **Hint:** Use `average="macro"` in `precision_score`, 
`recall_score`, and `f1_score` from scikit-learn.

## Installation
To run the notebook, you need to install the following 
dependencies:
```sh
pip install numpy pandas matplotlib scikit-learn tensorflow

