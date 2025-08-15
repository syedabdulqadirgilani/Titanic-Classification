# Titanic Survival Prediction — Easiest Version (Seaborn Dataset)

This mini project predicts whether a Titanic passenger **survived (1)** or **did not survive (0)** using a **very simple** Machine Learning model called **Logistic Regression**.

We use the **built‑in Titanic dataset from Seaborn**, so you **do not** need to download anything. The code is short, clean, and beginner‑friendly.

---

## 🎯 Purpose (Why are we doing this?)

- Learn the **basic steps** of a ML project: load data → clean data → encode → split → train → evaluate.  
- Build your **first classification model** (yes/no prediction).  
- Understand what **accuracy** means and how to read basic results.  

Use this project for your **portfolio, internship applications, and interviews**.

---

## 🧠 What you will learn

- How to **load a dataset** from Seaborn (`sns.load_dataset("titanic")`)  
- How to **select useful columns** (features)  
- How to **fill missing values** (imputation)  
- How to **convert text to numbers** (one‑hot encoding)  
- How to **split data** into train/test sets  
- How to **train Logistic Regression**  
- How to **check accuracy** and **(optionally)** show a confusion matrix  

---

## 📦 Requirements

Install these once:
```bash
pip install seaborn pandas scikit-learn matplotlib
```

---

## 🚀 Quick Start (Jupyter or .py file)

1. Copy your code into a new file, e.g. `titanic_seaborn_logreg.py`, **or** run it cell‑by‑cell in Jupyter/VS Code.  
2. Run the script:
```bash
python titanic_seaborn_logreg.py
```
No external files are needed — the dataset is included with Seaborn.

---

## 🗂️ Dataset (Seaborn’s Titanic)

- Loaded with: `df = sns.load_dataset("titanic")`  
- Important columns we use:
  - `survived` (0/1) → **target** (what we predict)
  - `pclass` (1,2,3) → passenger class (1 is best)
  - `sex` (male/female)
  - `age` (in years)
  - `sibsp` (siblings/spouses aboard)
  - `parch` (parents/children aboard)
  - `fare` (ticket price)
  - `embarked` (port: C/Q/S)

Note: Seaborn’s version is **slightly different** from Kaggle’s CSV (column names + values are already cleaned in many cases).

---

## 🧱 Full Code (Explained Step‑by‑Step)

```python
# %%
# -----------------------------------------
# Titanic Survival Prediction (Easiest Version)
# Using Seaborn's built-in Titanic dataset
# -----------------------------------------

# Import necessary libraries
import seaborn as sns  # for loading dataset and plotting
import pandas as pd    # for data handling
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.linear_model import LogisticRegression   # for prediction model
from sklearn.metrics import accuracy_score            # for model evaluation

# %%
# Step 1: Load Titanic dataset from Seaborn
df = sns.load_dataset("titanic")  # Titanic dataset comes built-in with Seaborn
df.head()  # Show first 5 rows to understand the data

# %%
# Step 2: Select useful columns for prediction
# We are picking only a few columns to make it simple
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# %%
# Step 3: Handle missing values
# Fill missing age with median age (middle value)
df["age"] = df["age"].fillna(df["age"].median())
# Fill missing embarked with most common value (mode)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# %%
# Step 4: Convert text to numbers (one-hot encoding)
# Example: "male" and "female" become binary (0/1) columns
# drop_first=True avoids duplicate columns (keeps it simple)
df = pd.get_dummies(df, drop_first=True)

# %%
# Step 5: Split data into features (X) and target (y)
X = df.drop("survived", axis=1)  # everything except 'survived'
y = df["survived"]               # only the 'survived' column

# %%
# Step 6: Train-test split
# 80% training, 20% testing (so we can check how well the model generalizes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Step 7: Create and train the model
# Logistic Regression is a simple, strong baseline for yes/no problems
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# %%
# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# %%
# Step 9: Check accuracy (how many correct predictions out of all)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)
```

---

## 📏 What does “Accuracy” mean?

**Accuracy** = (Number of correct predictions) ÷ (Total predictions)

- If accuracy = **0.80**, it means the model was correct **80%** of the time on the test data.  
- Accuracy is easy to understand, but it can be **misleading** if the classes are imbalanced (e.g., far more non‑survivors than survivors). For the Titanic dataset, accuracy is still a good starting point.

If you want a deeper check, add this (optional):

```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

- **Confusion Matrix** shows counts of:
  - True Negatives (correctly predicted 0)
  - False Positives (predicted 1 but actually 0)
  - False Negatives (predicted 0 but actually 1)
  - True Positives (correctly predicted 1)

- **Classification Report** adds **precision**, **recall**, and **F1‑score**, which are helpful if classes are imbalanced.

---

## 🧽 Why we did each preprocessing step

- **Fill missing `age` with median**: Models can’t handle missing values; median is a safe, robust guess.  
- **Fill missing `embarked` with most common value**: Simple, reasonable default.  
- **One‑hot encoding** with `pd.get_dummies(..., drop_first=True)`: Converts text to numbers so the model can use them; `drop_first=True` avoids redundant columns and keeps the model simpler.

---

## 🧪 Why train/test split?

We train on one part (**train set**) and test on a **separate** part (**test set**).  
This tells us how well the model will perform on **new, unseen** data.

---

## 🤖 Why Logistic Regression?

- Great **first choice** for binary (yes/no) problems.  
- **Fast**, **simple**, and **easy to explain** in interviews.  
- Outputs probabilities internally; we convert them to 0/1 labels for accuracy.

---

## 🛠 Troubleshooting

- **ConvergenceWarning**: Increase `max_iter` (e.g., `max_iter=1000`).  
- **ImportError**: Reinstall packages: `pip install seaborn pandas scikit-learn matplotlib`.  
- **Plots not showing**: If running in a script, make sure the script finishes and your environment supports GUI; in notebooks, they appear inline.

---

## 🌱 Next Steps (make it stronger)

- Add more features (e.g., `alone`, `who`, `adult_male` from Seaborn dataset).  
- Try models like **RandomForest** or **XGBoost**.  
- Use **cross‑validation** for better reliability.  
- Create a **PPT** with charts for interviews (I can generate one for you).

---

## 🗣️ How to explain this project in an interview

> “I built a simple Titanic survival predictor using Seaborn’s dataset.  
> I cleaned missing values, one‑hot encoded categories, split the data 80/20, and trained a Logistic Regression model.  
> I reported accuracy on the test set and, when needed, used a confusion matrix and classification report to discuss precision/recall.  
> The goal was to demonstrate an end‑to‑end ML pipeline in the simplest way possible.”

---

**Good luck — you’ve got this!** ✨
