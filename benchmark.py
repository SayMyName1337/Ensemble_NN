import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import time

# 1. Функции загрузки датасетов
def load_dataset(dataset_name):
    """Загрузка указанного датасета."""
    if dataset_name.lower() == 'iris':
        data = datasets.load_iris()
    elif dataset_name.lower() == 'wine':
        data = datasets.load_wine()
    elif dataset_name.lower() in ['breast', 'breast_cancer', 'cancer']:
        data = datasets.load_breast_cancer()
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    return X, y, feature_names, target_names

# 2. Функции добавления шума
def add_gaussian_noise(X, noise_level):
    """Добавление гауссовского шума в датасет."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def add_uniform_noise(X, noise_level):
    """Добавление равномерного шума в датасет."""
    noise = np.random.uniform(-noise_level, noise_level, X.shape)
    return X + noise

def add_impulse_noise(X, noise_level):
    """Добавление импульсного шума в датасет (соль и перец)."""
    X_noisy = X.copy()
    mask = np.random.rand(*X.shape) < noise_level
    X_noisy[mask] = np.random.uniform(X.min(), X.max(), size=np.sum(mask)).reshape(-1)
    return X_noisy

def add_missing_values(X, noise_level):
    """Добавление пропущенных значений в датасет (NaN)."""
    X_missing = X.copy()
    mask = np.random.rand(*X.shape) < noise_level
    X_missing[mask] = np.nan
    
    # Замена NaN средними значениями по столбцам для алгоритмов, 
    # которые не могут обрабатывать пропущенные значения
    col_mean = np.nanmean(X_missing, axis=0)
    inds = np.where(np.isnan(X_missing))
    X_missing[inds] = np.take(col_mean, inds[1])
    
    return X_missing

def add_noise(X, noise_type, noise_level):
    """Добавление шума указанного типа и уровня в датасет."""
    if noise_type == 'gaussian':
        return add_gaussian_noise(X, noise_level)
    elif noise_type == 'uniform':
        return add_uniform_noise(X, noise_level)
    elif noise_type == 'impulse':
        return add_impulse_noise(X, noise_level)
    elif noise_type == 'missing':
        return add_missing_values(X, noise_level)
    else:
        raise ValueError(f"Неизвестный тип шума: {noise_type}")

# 3. Определение моделей
def get_models():
    """Возвращает словарь моделей для оценки."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    return models

# 4. Функция оценки
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Обучение и оценка модели на заданных данных."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 5. Основная функция бенчмарка
def run_benchmark(dataset_name, noise_type, start_noise, end_noise, step_noise, num_experiments, noise_strategy='both'):
    """Запуск бенчмарка с указанными параметрами.
    
    Args:
        dataset_name: Имя датасета
        noise_type: Тип шума
        start_noise: Начальный уровень шума
        end_noise: Конечный уровень шума
        step_noise: Шаг изменения уровня шума
        num_experiments: Количество экспериментов для усреднения
        noise_strategy: Стратегия добавления шума ('both', 'test_only')
    """
    print(f"\nЗапуск бенчмарка на датасете {dataset_name} с шумом типа {noise_type}")
    print(f"Диапазон шума: от {start_noise} до {end_noise} с шагом {step_noise}")
    print(f"Количество экспериментов на уровень шума: {num_experiments}")
    print(f"Стратегия добавления шума: {'обе выборки' if noise_strategy == 'both' else 'только тестовая выборка'}")
    
    # Загрузка датасета
    X, y, feature_names, target_names = load_dataset(dataset_name)
    
    # Получение моделей
    models = get_models()
    
    # Подготовка хранилища результатов
    noise_levels = np.arange(start_noise, end_noise + step_noise/2, step_noise)
    results = {model_name: {
        'accuracy': np.zeros(len(noise_levels)),
        'precision': np.zeros(len(noise_levels)),
        'recall': np.zeros(len(noise_levels)),
        'f1': np.zeros(len(noise_levels))
    } for model_name in models.keys()}
    
    # Запуск экспериментов
    for i, noise_level in enumerate(noise_levels):
        print(f"\nУровень шума: {noise_level:.2f}")
        
        for exp in range(num_experiments):
            if noise_strategy == 'both':
                # Добавление шума во весь датасет
                X_noisy = add_noise(X, noise_type, noise_level)
                
                # Разделение данных
                X_train, X_test, y_train, y_test = train_test_split(
                    X_noisy, y, test_size=0.3, random_state=42+exp
                )
            else:  # 'test_only'
                # Разделение чистых данных
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42+exp
                )
                
                # Добавление шума только в тестовую выборку
                X_test = add_noise(X_test, noise_type, noise_level)
            
            # Масштабирование признаков
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Оценка каждой модели
            for model_name, model in models.items():
                metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
                
                # Накопление результатов
                for metric_name, metric_value in metrics.items():
                    results[model_name][metric_name][i] += metric_value / num_experiments
        
        # Вывод прогресса после каждого уровня шума
        for model_name in models.keys():
            print(f"{model_name}: Точность = {results[model_name]['accuracy'][i]:.4f}, "
                  f"F1 = {results[model_name]['f1'][i]:.4f}")
    
    return noise_levels, results

# 6. Визуализация и представление результатов
def plot_results(noise_levels, results, dataset_name, noise_type, metric='accuracy'):
    """Построение графика результатов бенчмарка."""
    plt.figure(figsize=(10, 6))
    
    for model_name, model_results in results.items():
        plt.plot(noise_levels, model_results[metric], marker='o', label=model_name)
    
    plt.xlabel('Уровень шума')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} vs. Уровень шума (датасет {dataset_name}, шум {noise_type})')
    plt.legend()
    plt.grid(True)
    
    # Сохранение графика
    filename = f"{dataset_name}_{noise_type}_{metric}.png"
    plt.savefig(filename)
    print(f"\nГрафик результатов сохранен как {filename}")
    
    # Отображение графика в интерактивном режиме
    plt.show()

def create_summary_table(noise_levels, results, dataset_name, noise_type):
    """Создание и вывод сводной таблицы результатов."""
    # Создаем пустой DataFrame для хранения результатов
    all_data = []
    
    # Заполняем данные из результатов
    for noise_idx, noise_level in enumerate(noise_levels):
        for model_name, model_results in results.items():
            row_data = {
                'Датасет': dataset_name,
                'Тип шума': noise_type,
                'Уровень шума': f"{noise_level:.2f}",
                'Модель': model_name,
                'Точность (Accuracy)': f"{model_results['accuracy'][noise_idx]:.4f}",
                'Полнота (Recall)': f"{model_results['recall'][noise_idx]:.4f}",
                'Точность (Precision)': f"{model_results['precision'][noise_idx]:.4f}",
                'F1-мера': f"{model_results['f1'][noise_idx]:.4f}"
            }
            all_data.append(row_data)
    
    # Создаем DataFrame
    df = pd.DataFrame(all_data)
    
    # Выводим таблицу в консоль
    print("\n" + "=" * 120)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 120)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(df)
    print("=" * 120)
    
    # Создаем сводную таблицу по моделям (усреднение по всем уровням шума)
    print("\n" + "=" * 120)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (среднее по всем уровням шума)")
    print("=" * 120)
    
    model_summary = []
    for model_name, model_results in results.items():
        model_summary.append({
            'Модель': model_name,
            'Средняя точность (Accuracy)': f"{np.mean(model_results['accuracy']):.4f}",
            'Средняя полнота (Recall)': f"{np.mean(model_results['recall']):.4f}",
            'Средняя точность (Precision)': f"{np.mean(model_results['precision']):.4f}",
            'Средняя F1-мера': f"{np.mean(model_results['f1']):.4f}"
        })
    
    model_df = pd.DataFrame(model_summary)
    print(model_df)
    print("=" * 120)
    
    # Сохраняем таблицу в CSV
    table_filename = f"{dataset_name}_{noise_type}_table.csv"
    df.to_csv(table_filename, index=False)
    print(f"\nСводная таблица сохранена в {table_filename}")
    
    return df

# 7. Консольный интерфейс
def main():
    print("=" * 80)
    print("Бенчмарк алгоритмов машинного обучения с шумом")
    print("=" * 80)
    
    # Выбор датасета
    print("\nДоступные датасеты:")
    print("1. Iris (Ирисы)")
    print("2. Wine (Вино)")
    print("3. Breast Cancer (Рак груди)")
    
    dataset_choice = input("\nВыберите датасет (1-3): ")
    if dataset_choice == '1':
        dataset_name = 'iris'
    elif dataset_choice == '2':
        dataset_name = 'wine'
    elif dataset_choice == '3':
        dataset_name = 'breast_cancer'
    else:
        print("Неверный выбор. Используется датасет Iris по умолчанию.")
        dataset_name = 'iris'
    
    # Выбор типа шума
    print("\nДоступные типы шума:")
    print("1. Gaussian (Гауссовский)")
    print("2. Uniform (Равномерный)")
    print("3. Impulse (Импульсный)")
    print("4. Missing Values (Пропущенные значения)")
    
    noise_choice = input("\nВыберите тип шума (1-4): ")
    if noise_choice == '1':
        noise_type = 'gaussian'
    elif noise_choice == '2':
        noise_type = 'uniform'
    elif noise_choice == '3':
        noise_type = 'impulse'
    elif noise_choice == '4':
        noise_type = 'missing'
    else:
        print("Неверный выбор. Используется Гауссовский шум по умолчанию.")
        noise_type = 'gaussian'
        
    # Выбор стратегии добавления шума
    print("\nСтратегия добавления шума:")
    print("1. Добавлять шум в обе выборки (обучающую и тестовую)")
    print("2. Добавлять шум только в тестовую выборку (обучение на чистых данных)")
    
    strategy_choice = input("\nВыберите стратегию (1-2): ")
    if strategy_choice == '1':
        noise_strategy = 'both'
    elif strategy_choice == '2':
        noise_strategy = 'test_only'
    else:
        print("Неверный выбор. Используется стратегия добавления шума в обе выборки.")
        noise_strategy = 'both'
    
    # Параметры шума
    try:
        start_noise = float(input("\nВведите начальный уровень шума (например, 0.0): "))
        end_noise = float(input("Введите конечный уровень шума (например, 1.0): "))
        step_noise = float(input("Введите шаг уровня шума (например, 0.1): "))
        num_experiments = int(input("Введите количество экспериментов на уровень шума (например, 5): "))
    except ValueError:
        print("Неверный ввод. Используются значения по умолчанию.")
        start_noise = 0.0
        end_noise = 1.0
        step_noise = 0.1
        num_experiments = 5
    
    # Запуск бенчмарка
    start_time = time.time()
    noise_levels, results = run_benchmark(
        dataset_name, noise_type, start_noise, end_noise, step_noise, num_experiments, noise_strategy
    )
    end_time = time.time()
    
    print(f"\nБенчмарк завершен за {end_time - start_time:.2f} секунд.")
    
    # Построение графиков
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_results(noise_levels, results, dataset_name, noise_type, metric)
    
    # Создание сводной таблицы
    summary_df = create_summary_table(noise_levels, results, dataset_name, noise_type)
    
    # Сохранение результатов в CSV
    results_df = pd.DataFrame()
    for model_name, model_results in results.items():
        for metric_name, metric_values in model_results.items():
            col_name = f"{model_name}_{metric_name}"
            results_df[col_name] = metric_values
    
    results_df['noise_level'] = noise_levels
    filename = f"{dataset_name}_{noise_type}_results.csv"
    results_df.to_csv(filename, index=False)
    print(f"Результаты сохранены в {filename}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nБенчмарк прерван пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)