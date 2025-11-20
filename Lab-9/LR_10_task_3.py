import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

# 1. Завантаження та підготовка даних
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

# 2. Кодування рядкових даних (Label Encoding)
label_encoder = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 3. Розбивка даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# 4. Навчання регресора
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# 5. Оцінка ефективності
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# 6. Тестування на одиничному прикладі
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0

for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(item)
    else:
        # Використовуємо збережені кодувальники
        # Обробка випадку, якщо значення не зустрічалося раніше (хоча в цьому прикладі все ок)
        try:
            test_datapoint_encoded[i] = int(label_encoder[count].transform([item])[0])
        except ValueError:
             print(f"Label {item} not seen in training data")
        count += 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогноз
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))