# 🎓 Student Performance Predictor 📊

This project uses **Linear Regression** to predict a student's performance based on various factors like study hours, previous scores, extracurricular activities, sleep hours, and number of practice papers attempted.

---

## 📁 Dataset

✅ 1000 rows of student data  
✅ Features used for training:

- `Hours Studied` 📚  
- `Previous Scores` 📝  
- `Extracurricular Activities` 🎨  
- `Sleep Hours` 😴  
- `Sample Question Papers Practiced` 🧠  

---

## 🧠 Tech Stack

- Python 🐍  
- NumPy 🔢  
- Scikit-learn 🤖  
- Jupyter Notebook 📓  

---

## 🛠️ Model Training

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📥 Load your dataset
data = pd.read_csv("student_data.csv")

# 🎯 Features and Target
X = data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]]
y = data["Final Score"]

# ✂️ Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# 🔍 Predictions
y_pred = lm.predict(X_test)

# 📈 Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ MAE:", mae)
print("✅ MSE:", mse)
print("✅ RMSE:", rmse)
print("✅ R-squared:", r2)
