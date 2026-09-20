# 🚗 Ford Car Price Prediction using Linear Regression

## 📌 Project Overview

This project focuses on predicting the **price of Ford cars** using **Linear Regression**, a supervised machine learning algorithm.

The project uses a Ford car dataset containing **17,966 records** and 9 columns. The analysis explores how different car features such as **model, year, mileage, transmission, fuel type, tax, MPG, and engine size** are related to car prices.

The complete project is implemented in **Python using Jupyter Notebook**, with data analysis, visualization, preprocessing, feature encoding, feature scaling, model training, and evaluation.

---

## 🎯 Objective

The main objective of this project is to:

* Understand the factors affecting Ford car prices.
* Perform Exploratory Data Analysis (EDA).
* Prepare categorical and numerical features for machine learning.
* Build a Linear Regression model.
* Predict car prices based on available car features.
* Evaluate the performance of the regression model using R² and Adjusted R².

---

## 📊 Dataset

The dataset contains **17,966 Ford car records** with the following 9 columns:

| Feature        | Description                   |
| -------------- | ----------------------------- |
| `model`        | Ford car model                |
| `year`         | Manufacturing year            |
| `price`        | Car price — target variable   |
| `transmission` | Type of transmission          |
| `mileage`      | Distance travelled by the car |
| `fuelType`     | Type of fuel used             |
| `tax`          | Vehicle tax                   |
| `mpg`          | Miles per gallon              |
| `engineSize`   | Engine size                   |

### Dataset Information

* **Total records:** 17,966
* **Total columns:** 9
* **Missing values:** No missing values were found in the dataset.
* **Target variable:** `price`

---

# 🔎 Exploratory Data Analysis

Before building the machine learning model, Exploratory Data Analysis was performed to understand the dataset and relationships between variables.

### Visualizations used:

* Price distribution using Histogram
* Correlation Heatmap
* Year vs Price Boxplot
* Mileage vs Price Scatter Plot
* Engine Size vs Price Boxplot
* Transmission vs Price Boxplot
* Fuel Type vs Price Boxplot
* Model vs Price Boxplot

These visualizations helped understand the relationship between car characteristics and their prices.

---

# 🧹 Data Preprocessing

The dataset contains both **numerical and categorical features**, so preprocessing was performed before training the model.

### 1. Separating Features and Target

The target variable was:

```python
Y = df['price']
```

The remaining columns were used as input features:

```python
X = df.drop(columns=['price'], axis=1)
```

So:

* **X → Input features**
* **Y → Car price / target**

---

### 2. Categorical Encoding

The categorical columns were:

```text
model
transmission
fuelType
```

One-Hot Encoding was applied using:

```python
pd.get_dummies()
```

with `drop_first=True`.

This converts categorical values into numerical features that can be used by the Linear Regression model.

---

### 3. Label Encoding

A second preprocessing approach was also explored using `LabelEncoder` for:

```text
model
transmission
fuelType
```

This created another encoded dataset for comparison.

---

### 4. Feature Scaling

Standardization was applied using `StandardScaler`.

Numerical features included:

```text
year
mileage
tax
mpg
engineSize
```

Standard scaling transforms features so that they are represented on a comparable scale.

---

# 🤖 Machine Learning Model

## Linear Regression

Linear Regression was used because this is a **regression problem**, where the goal is to predict a continuous numerical value — the **price of a car**.

The basic idea of Linear Regression is:

```text
Predicted Price = Intercept + (Coefficient × Feature)
```

For multiple features, the model learns the contribution of multiple variables to the predicted price.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing data using:

```python
train_test_split()
```

The configuration used was:

```python
test_size = 0.33
random_state = 42
```

Approximately:

* **67% → Training data**
* **33% → Testing data**

The model learns patterns from the training data and is then evaluated on unseen testing data.

---

# 📈 Model Training

The Linear Regression model was created using:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

After training, predictions were generated using:

```python
y_pred = model.predict(X_test)
```

---

# 📊 Model Evaluation

The project evaluated the model using **R² Score** and **Adjusted R² Score**.

## R² Score

For the model using One-Hot Encoding, the obtained R² score was:

```text
R² Score = 0.8397
```

This indicates that the model explains approximately **83.97% of the variation in the test-set car prices** for this particular split.

---

## Adjusted R² Score

The obtained Adjusted R² score was:

```text
Adjusted R² = 0.8387
```

Adjusted R² takes the number of predictors into account and adjusts the R² value accordingly.

---

## 🔄 Encoding Comparison

The notebook also experimented with a second preprocessing approach using Label Encoding.

For that model, the recorded R² score was:

```text
R² Score = 0.7310
```

This shows that the preprocessing/encoding approach had a noticeable effect on the model's measured performance.

---

# 🛠️ Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Machine Learning
* **Linear Regression** — Regression algorithm

---

# 📁 Project Structure

```text
Linear-Regression/
│
├── ford1.ipynb
├── ford.csv
└── README.md
```

> `ford.csv` is the dataset used by the notebook. Make sure you have the required dataset available in the same directory before running the notebook.

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-project.git
```

### 2. Navigate to the project directory

```bash
cd linear-regression-project
```

### 3. Install required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
ford1.ipynb
```

Run the cells from top to bottom.

---

# 💡 Key Learnings

Through this project, I learned how to:

* Load and inspect a real-world dataset.
* Identify numerical and categorical variables.
* Check for missing values.
* Perform Exploratory Data Analysis.
* Visualize relationships between features and target variables.
* Encode categorical variables.
* Scale numerical features.
* Split data into training and testing sets.
* Train a Linear Regression model.
* Generate predictions.
* Evaluate regression models using R² and Adjusted R².
* Compare the effect of different feature encoding approaches.

---

# 📌 Conclusion

This project demonstrates an end-to-end **Machine Learning regression workflow** using a Ford car dataset.

Starting from raw data, the project goes through:

```text
Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning / Checking
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
Train-Test Split
      ↓
Linear Regression
      ↓
Prediction
      ↓
Model Evaluation
```

The Linear Regression model achieved an **R² score of approximately 0.84** with the One-Hot Encoded feature set on the evaluated test split.

This project helped demonstrate practical understanding of the complete machine learning pipeline, from **data exploration to model evaluation**.

---

## 👩‍💻 Author

**Sneha Kashyap**

B.Tech Graduate | Python | Data Analysis | Machine Learning

GitHub: [https://github.com/snehaa94]
