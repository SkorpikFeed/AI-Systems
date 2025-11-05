import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Завантаження даних 
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Поділ даних (50% на тест, random_state=0) 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# 3. Створення та навчання моделі 
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# 4. Прогноз 
ypred = regr.predict(Xtest)

# 5. Розрахунок метрик 
print("--- Метрики якості (Завдання 8.1) ---")
print("Коефіцієнти (regr.coef_): \n", regr.coef_)
print("\nПеретин (regr.intercept_):", regr.intercept_)
print("\nR2 score: %.2f" % r2_score(ytest, ypred))
print("Mean absolute error (MAE): %.2f" % mean_absolute_error(ytest, ypred))
print("Mean squared error (MSE): %.2f" % mean_squared_error(ytest, ypred))

# 6. Побудова графіка 
fig, ax = plt.subplots()
# "Виміряно" (ytest) по осі X, "Передбачено" (ypred) по осі Y
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
# Ідеальна лінія, де y=x 
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4) 
ax.set_xlabel('Виміряно (Справжні значення)')
ax.set_ylabel('Передбачено (Прогнози моделі)')
plt.title('Завдання 8.1: Diabetes Dataset (Виміряно vs Передбачено)')
plt.show()