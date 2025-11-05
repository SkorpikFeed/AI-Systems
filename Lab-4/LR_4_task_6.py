import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

try:
    # 1. Завантаження даних
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    print("--- 1. Розрахунок метрик для SVM (Машина Опорних Векторів) ---")

    # 2. Створення класифікатора SVM
    # Використовуємо 'kernel='rbf' (ядро Гауса) - стандартний
    # та ефективний вибір для більшості завдань
    classifier_svm = svm.SVC(kernel='rbf') 

    # 3. Розрахунок показників якості (3-fold cross-validation)
    num_folds = 3

    accuracy_values_svm = cross_val_score(classifier_svm,
                                          X, y, scoring='accuracy', cv=num_folds)
    print(f"SVM Accuracy: {round(100 * accuracy_values_svm.mean(), 2)}%")

    precision_values_svm = cross_val_score(classifier_svm,
                                           X, y, scoring='precision_weighted', cv=num_folds)
    print(f"SVM Precision: {round(100 * precision_values_svm.mean(), 2)}%")

    recall_values_svm = cross_val_score(classifier_svm,
                                        X, y, scoring='recall_weighted', cv=num_folds)
    print(f"SVM Recall: {round(100 * recall_values_svm.mean(), 2)}%")

    f1_values_svm = cross_val_score(classifier_svm,
                                    X, y, scoring='f1_weighted', cv=num_folds)
    print(f"SVM F1: {round(100 * f1_values_svm.mean(), 2)}%")

    print("\n--- 2. Для порівняння: Метрики Наївного Баєса (з Завдання 2.4) ---")
    
    # Розрахунок метрик для Naive Bayes для прямого порівняння
    classifier_nb = GaussianNB()
    
    accuracy_values_nb = cross_val_score(classifier_nb,
                                          X, y, scoring='accuracy', cv=num_folds)
    print(f"Naive Bayes Accuracy: {round(100 * accuracy_values_nb.mean(), 2)}%")

    precision_values_nb = cross_val_score(classifier_nb,
                                           X, y, scoring='precision_weighted', cv=num_folds)
    print(f"Naive Bayes Precision: {round(100 * precision_values_nb.mean(), 2)}%")

    recall_values_nb = cross_val_score(classifier_nb,
                                        X, y, scoring='recall_weighted', cv=num_folds)
    print(f"Naive Bayes Recall: {round(100 * recall_values_nb.mean(), 2)}%")

    f1_values_nb = cross_val_score(classifier_nb,
                                    X, y, scoring='f1_weighted', cv=num_folds)
    print(f"Naive Bayes F1: {round(100 * f1_values_nb.mean(), 2)}%")

except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{input_file}' не знайдено.")
    print("Будь ласка, додайте файл у папку зі скриптом.")
except Exception as e:
    print(f"Виникла помилка: {e}")