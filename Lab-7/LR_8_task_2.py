import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- Завдання 8.2: Генерація даних та Побудова моделей ---

# 1. Генерація даних (Варіант 1) 
np.random.seed(42) # Для відтворюваності результатів
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
y = y.ravel() # Перетворюємо y на (m,) для сумісності з fit

# 2. Побудова Лінійної регресії (ступінь 1)
lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X, y)
X_new = np.linspace(-5, 1, 100).reshape(100, 1)
y_new_simple = lin_reg_simple.predict(X_new)

# 3. Побудова Поліноміальної регресії (ступінь 2) 
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# print("Оригінальна ознака X[0]:", X[0])
# print("Поліноміальні ознаки X_poly[0]:", X_poly[0])

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y) # 
y_new_poly = lin_reg_poly.predict(poly_features.transform(X_new))

# 4. Виведення коефіцієнтів (Завдання 8.2) 
print("--- Коефіцієнти (Завдання 8.2) ---")
print("Оригінальна модель: y = 0.5*X^2 + 1.0*X + 2.0 + шум")
# Коефіцієнти йдуть у порядку [X, X^2]
print(f"Отримана модель: y = {lin_reg_poly.coef_[1]:.2f}*X^2 + {lin_reg_poly.coef_[0]:.2f}*X + {lin_reg_poly.intercept_:.2f}")

# 5. Графік (Завдання 8.2) 
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вихідні дані (Варіант 1)', s=10)
plt.plot(X_new, y_new_simple, "r-", linewidth=2, label="Лінійна регресія (ступінь 1)")
plt.plot(X_new, y_new_poly, "g-", linewidth=3, label="Поліноміальна регресія (ступінь 2)")
plt.xlabel("$x_1$")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 8.2: Лінійна vs Поліноміальна регресія")
plt.axis([-5, 1, 0, 10])
plt.show()


# --- Завдання 8.3: Криві навчання ---

# 1. Функція для побудови кривих навчання 
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.legend()
    # Встановлюємо межу Y, щоб графіки були читабельними
    plt.ylim(0, 3) 

# 2. Графік 1: Лінійна модель (недонавчання) 
lin_reg = LinearRegression()
plt.figure(figsize=(10, 6))
plot_learning_curves(lin_reg, X, y)
plt.title("Завдання 8.3: Криві навчання (Лінійна модель, ступінь 1)")
plt.show()

# 3. Графік 2: Поліноміальна модель (ступінь 10, перенавчання) 
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plt.figure(figsize=(10, 6))
plot_learning_curves(polynomial_regression_10, X, y)
plt.title("Завдання 8.3: Криві навчання (Поліноміальна, ступінь 10)")
plt.show()

# 4. Графік 3: Поліноміальна модель (ступінь 2, "золота середина")
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plt.figure(figsize=(10, 6))
plot_learning_curves(polynomial_regression_2, X, y)
plt.title("Завдання 8.3: Криві навчання (Поліноміальна, ступінь 2)")
plt.show()