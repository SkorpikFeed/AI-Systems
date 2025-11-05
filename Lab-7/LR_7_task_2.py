import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Вхідний файл, який містить дані (Варіант 1) [cite: 496]
input_file = 'data_regr_1.txt' 

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори (80% / 20%)
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створення об'єкта лінійного регресора
regressor = linear_model.LinearRegression()

# Навчання моделі
regressor.fit(X_train, y_train)

# Прогнозування результату
y_test_pred = regressor.predict(X_test)

# Побудова графіка
plt.scatter(X_test, y_test, color='blue', label='Тестові дані (Вар. 1)')
plt.plot(X_test, y_test_pred, color='red', linewidth=4, label='Лінія регресії (Вар. 1)')
plt.title('Завдання 7.2: Лінійна регресія (Варіант 1)')
plt.xlabel('Ознака')
plt.ylabel('Цільова змінна')
plt.legend()
plt.show()

# Оцінка якості [cite: 504]
print("--- Linear regressor performance (Завдання 7.2) ---")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))