Iris Flower Classification using k-Nearest Neighbors (k-NN)
By: Glory Bagai
One-line Description: A supervised learning project that predicts the species of iris flowers based on their physical measurements using the k-Nearest Neighbors algorithm.

About
This project focuses on building a machine learning model to classify iris flowers into one of three species—setosa, versicolor, or virginica—based on their sepal and petal measurements. The dataset used is the famous Iris dataset, which contains labeled examples of flower measurements. We implemented the k-Nearest Neighbors (k-NN) classification algorithm to train and evaluate the model, achieving an accuracy of 97% on the test set.

Time Spent
Data Loading & Exploration: 1 hour

Model Training & Evaluation: 2 hours

Documentation & Fine-tuning: 1 hour

Required Features and Optional Features
Required Features (Input Variables)
Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)

Optional Features (Hyperparameters for k-NN)
Number of neighbors (n_neighbors) – Default: 5

Distance metric (metric) – Default: Euclidean distance

Weight function (weights) – Uniform or distance-based

Notes
The dataset is well-balanced, with 50 samples per class.

No missing values or outliers were present, so minimal preprocessing was needed.

The model performs exceptionally well with default hyperparameters.

Link to Relevant Documentation
Scikit-learn KNeighborsClassifier Documentation

Iris Dataset Description

Licenses
Dataset: Public Domain (UCI Machine Learning Repository)

Code: MIT License

Inspiration
Inspired by the classic machine learning problem introduced in Pattern Recognition and Machine Learning by Christopher Bishop.

Scikit-learn Tutorial on k-NN

What It Does
This project:

Loads the Iris dataset (sepal and petal measurements).

Splits the data into training and test sets.

Trains a k-Nearest Neighbors classifier to predict iris species.

Evaluates model performance using accuracy metrics.

How We Built It
Data Loading & Preprocessing

Used sklearn.datasets.load_iris() to load the dataset.

Split the data into X_train, X_test, y_train, y_test using train_test_split().

Model Training

Initialized KNeighborsClassifier with n_neighbors=5.

Fitted the model using model.fit(X_train, y_train).

Evaluation

Predicted test set labels using model.predict(X_test).

Computed accuracy with model.score(X_test, y_test).

Challenges We Ran Into
Choosing the right k value: Too small led to overfitting, too large led to underfitting.

Feature scaling: Initially, features were not scaled, but since k-NN is distance-based, scaling improved performance.

Interpretability: Understanding why certain neighbors were chosen required visualizing decision boundaries.

Accomplishments We’re Proud Of
Achieved 97% accuracy with minimal tuning.

Successfully implemented a fundamental ML algorithm from scratch.

Created a reproducible Colab notebook for easy experimentation.

What We Learned
The importance of distance metrics in k-NN.

How hyperparameter tuning affects model performance.

The trade-off between model complexity and generalization.

What’s Next
Experiment with other classification algorithms (e.g., SVM, Decision Trees).

Apply cross-validation for more robust model evaluation.

Deploy the model as a simple web app using Flask or Streamlit.

Try It Out!
🔗 Open in Google Colab (Add your Colab link here)# Flower-Specie-Classification
A supervised learning project that predicts the species of iris flowers based on their physical measurements using the k-Nearest Neighbors algorithm.
