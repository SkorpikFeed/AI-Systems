import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# 1. Завантаження даних
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбиття на класи для візуалізації
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Візуалізація
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Вхідні дані (Дисбаланс)')

# 2. Розбивка даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# 3. Налаштування параметрів (з балансуванням чи без)
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params['class_weight'] = 'balanced'
        print("Режим: З балансуванням класів")
    else:
        print("Режим: Без балансування (ігнорування аргументу)")
else:
    print("Режим: Без балансування")

# 4. Навчання класифікатора
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# 5. Прогноз і звіт
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on test dataset\n")
# zero_division=0 приховує попередження, якщо клас не передбачено жодного разу
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
print("#"*40 + "\n")

plt.show()