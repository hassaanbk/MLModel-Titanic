## Titanic Survival Prediction using Logistic Regression

This Python script is designed for predicting survival outcomes for passengers aboard the Titanic using Logistic Regression. The code performs data preprocessing, feature engineering, model training, and evaluation on the Titanic dataset (`titanic3.csv`), which should be located in the specified file path: `/Users/Hassaan/Desktop/Data Warehousing and Predictive Analytics/Exercise 10`.

### Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- scikit-learn

You can install these libraries using `pip`:

```bash
pip install numpy pandas scikit-learn
```

### Usage

1. **Data Preparation**: Ensure that the Titanic dataset (`titanic3.csv`) is in the specified file location.

2. **Running the Code**: Execute the script in a Python environment (e.g., Jupyter Notebook, Spyder). The code performs the following tasks:

   - Load the Titanic dataset.
   - Preprocess and clean the data by handling missing values.
   - Perform one-hot encoding for categorical features.
   - Standardize numerical features.
   - Train a Logistic Regression model to predict survival.
   - Evaluate the model using 10-fold cross-validation and calculate accuracy.
   - Display a confusion matrix to assess model performance.

3. **Model Export**: The trained Logistic Regression model is saved as `model_lr2.pkl`, and the list of model columns is saved as `model_columns.pkl` in the same directory as the script.

### Code Structure

The code is divided into the following sections:

- Importing necessary libraries and loading the dataset.
- Data exploration and preprocessing, including handling missing values and one-hot encoding.
- Standardizing numerical features.
- Splitting the data into training and testing sets.
- Training a Logistic Regression model.
- Evaluating the model using 10-fold cross-validation and accuracy metrics.
- Saving the trained model and model columns for future use.

### Results

The code provides an accuracy score for the Logistic Regression model and a confusion matrix to assess the model's performance in predicting survival on the Titanic dataset.

### Customization

Feel free to modify and customize the code to fit your specific needs. You can experiment with different machine learning algorithms, hyperparameters, or preprocessing techniques to improve model performance.

For any questions or issues, please contact [Hassaan](mailto:hassaan@email.com).

Enjoy working with the Titanic dataset and predicting passenger survival using Logistic Regression!
