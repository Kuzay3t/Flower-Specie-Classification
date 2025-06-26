🌸 Iris Species Classification using K-Nearest Neighbors
By:
Glory Bagai

📘 About
This project is a machine learning classification task that predicts the species of an iris flower based on its physical measurements. By using the classic Iris dataset, we built a supervised learning model with the K-Nearest Neighbors (KNN) algorithm to classify the flower as either Setosa, Versicolor, or Virginica. The simplicity and interpretability of KNN made it an ideal choice for this beginner-friendly classification problem.

⏱ Time Spent
Total: Approximately 4–6 hours

Data exploration & preprocessing: 1.5 hours

Model building & training: 1 hour

Evaluation & accuracy testing: 1 hour

Documentation, polishing, and debugging: 1.5–2 hours

🧰 Required Features
Use of Scikit-learn’s KNeighborsClassifier

Supervised learning approach

Use of fit() and score() methods

3-class classification for Setosa, Versicolor, and Virginica

Evaluation of accuracy on test set

Use of train-test split (80%-20%)

💡 Optional Features (Implemented)
Accuracy computation (~97%)

Inline plotting for data exploration (e.g., matplotlib, seaborn)

Commented code cells in Colab

Code modularization into functions

📝 Notes
Dataset used: Iris Dataset from Scikit-learn's built-in datasets.

The task is supervised learning, as the training data includes labeled examples.

It is a multi-class classification problem (3 classes).

Our model was able to achieve an accuracy of approximately 97% on unseen test data.

🔗 Link to Relevant Documentation
Scikit-learn - KNeighborsClassifier

Iris Dataset Info (Wikipedia)

Google Colab Documentation

🪪 Licenses
This project uses:

MIT License – Open for use, modification, and distribution.
View License

🌟 Inspiration
This project was inspired by the classic Iris flower classification task made famous by Ronald Fisher in 1936.
🔗 Original Iris Dataset Overview

⚙️ What It Does
Predicts the species of an iris flower using physical measurements including:

Sepal length

Sepal width

Petal length

Petal width

It applies the K-Nearest Neighbors algorithm to analyze training data and classify a new sample based on the most common label among its nearest neighbors.

🛠 How We Built It
Data Loading
Used Scikit-learn’s load_iris() function.

Data Preprocessing
Split the dataset into training and testing sets using train_test_split().

Model Building
Instantiated KNeighborsClassifier with default parameters and trained using the fit() method.

Evaluation
Evaluated the model with the score() method and achieved ~97% accuracy.

Visualization (Optional)
Explored data patterns with Seaborn and Matplotlib plots.

🧗 Challenges We Ran Into
Choosing the right value of k (number of neighbors) for optimal performance.

Understanding how different features affect classification.

Ensuring the model generalizes well without overfitting.

🏆 Accomplishments We’re Proud Of
Successfully implemented a model with 97% accuracy.

Gained hands-on experience in end-to-end machine learning workflow.

Understood the intuition and working of KNN classification.

📚 What We Learned
How to load and explore a standard dataset in Scikit-learn.

The fundamentals of supervised learning and classification tasks.

The workings of K-Nearest Neighbors and how predictions are made.

Evaluation of ML models using test data.

🚀 What’s Next
Experiment with other classification algorithms like SVM, Decision Trees, or Random Forests.

Tune KNN parameters like number of neighbors (k) and distance metrics.

Build a simple web app to classify iris species in real-time using Streamlit or Flask.
