'''
- Улучшена архитектура нейронной сети
- Улучшена предобработка данных
- Улучшена устойчивость к шуму
- Улучшена оптимизаиця, время экспеирмента стало меньше
- Улучшенная архитектура нейронной сети
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, Activation, GaussianNoise
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import os
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import joblib
import xgboost as xgb
import lightgbm as lgb
from imblearn.combine import SMOTETomek

# Отключение предупреждений для более чистого вывода
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

@tf.keras.utils.register_keras_serializable(package="Custom", name="FocalLoss")
class FocalLoss(tf.keras.losses.Loss):
    """Реализация Focal Loss для бинарной классификации"""
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom", name="CategoricalFocalLoss")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Реализация Focal Loss для многоклассовой классификации"""
    
    def __init__(self, gamma=2.0, **kwargs):
        super(CategoricalFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, self.gamma) * y_true
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    
    def get_config(self):
        config = super(CategoricalFocalLoss, self).get_config()
        config.update({
            'gamma': self.gamma
        })
        return config

class NoiseInjector:
    """Класс для добавления различных типов шума в данные"""
    
    @staticmethod
    def add_gaussian_noise(X, intensity):
        """Добавляет гауссовский шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (стандартное отклонение)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = np.random.normal(0, intensity, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    @staticmethod
    def add_uniform_noise(X, intensity):
        """Добавляет равномерный шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (максимальная амплитуда)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = np.random.uniform(-intensity, intensity, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    @staticmethod
    def add_impulse_noise(X, intensity):
        """Добавляет импульсный шум к данным (случайные выбросы)
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность выброса)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < intensity
        
        # Создаем импульсы с крайними значениями
        impulses = np.random.choice([-5, 5], size=X.shape)
        X_noisy[mask] = impulses[mask]
        
        return X_noisy
    
    @staticmethod
    def add_missing_values(X, intensity):
        """Добавляет пропущенные значения к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность пропуска)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < intensity
        X_noisy[mask] = np.nan
        
        return X_noisy
    
    @staticmethod
    def add_salt_pepper_noise(X, intensity):
        """Добавляет шум типа "соль и перец" к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность искажения)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        
        # Маска для "соли" (максимальные значения)
        salt_mask = np.random.random(X.shape) < intensity/2
        X_noisy[salt_mask] = np.max(X)
        
        # Маска для "перца" (минимальные значения)
        pepper_mask = np.random.random(X.shape) < intensity/2
        X_noisy[pepper_mask] = np.min(X)
        
        return X_noisy
    
    @staticmethod
    def add_multiplicative_noise(X, intensity):
        """Добавляет мультипликативный шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = 1 + np.random.normal(0, intensity, X.shape)
        X_noisy = X * noise
        return X_noisy

class NoisePreprocessor:
    """Улучшенный класс для предобработки зашумленных данных с учетом типа шума"""
    
    def __init__(self):
        """Инициализирует препроцессор данных"""
        self.preprocessors = {}
        # Словарь для кэширования характеристик данных для разных наборов
        self.dataset_stats = {}
        
    def preprocess_gaussian_noise(self, X):
        """Улучшенная предобработка данных с гауссовским шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        from scipy.ndimage import median_filter, gaussian_filter
        from scipy.signal import wiener
        
        X_processed = X.copy()
        
        # Оцениваем уровень шума для каждого признака
        noise_levels = np.std(X, axis=0)
        
        # Обрабатываем каждый признак отдельно
        for i in range(X.shape[1]):
            feature = X[:, i].reshape(-1, 1)
            noise_level = noise_levels[i]
            
            # Выбираем метод в зависимости от уровня шума
            if noise_level > 0.5:  # Сильный шум
                # Медианный фильтр для сильного шума
                # Преобразуем фильтр для работы с 1D данными
                feature_1d = feature.flatten()
                filtered = median_filter(feature_1d, size=3)
                X_processed[:, i] = filtered
            elif noise_level > 0.2:  # Средний шум
                # Фильтр Винера адаптивно убирает шум, сохраняя детали
                feature_1d = feature.flatten()
                filtered = wiener(feature_1d, mysize=3)
                X_processed[:, i] = filtered
            else:  # Слабый шум
                # Гауссовский фильтр для слабого шума
                feature_1d = feature.flatten()
                filtered = gaussian_filter(feature_1d, sigma=1)
                X_processed[:, i] = filtered
        
        # Дополнительно выполняем Singular Spectrum Analysis (SSA)
        # для более эффективного удаления шума, сохраняя структуру данных
        if X.shape[0] > 10:  # Требуется достаточно точек для SSA
            try:
                X_processed = self._apply_ssa(X_processed, window_size=min(10, X.shape[0] // 3))
            except:
                # Если SSA не сработал, оставляем предыдущую обработку
                pass
                
        return X_processed
    
    def _apply_ssa(self, X, window_size=3, n_components=None):
        """Применяет Singular Spectrum Analysis для сглаживания временных рядов
        
        Args:
            X: Исходные данные
            window_size: Размер окна для SSA
            n_components: Количество компонент (если None, выбирается автоматически)
            
        Returns:
            X_smoothed: Сглаженные данные
        """
        from sklearn.decomposition import TruncatedSVD
        
        X_smoothed = X.copy()
        
        # Обрабатываем каждый признак отдельно
        for i in range(X.shape[1]):
            feature = X[:, i]
            
            # Создаем траекторную матрицу (embedding)
            K = X.shape[0] - window_size + 1
            trajectory = np.zeros((K, window_size))
            
            for j in range(K):
                trajectory[j, :] = feature[j:j+window_size]
            
            # Определяем количество компонент (если не указано)
            if n_components is None:
                # Используем 80% объясненной дисперсии
                svd = TruncatedSVD(n_components=min(window_size-1, K-1))
                svd.fit(trajectory)
                explained_variance_ratio = svd.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                n_components = np.argmax(cumulative_variance >= 0.8) + 1
                n_components = max(1, min(n_components, window_size // 2))
            
            # Выполняем SVD с выбранным количеством компонент
            svd = TruncatedSVD(n_components=n_components)
            trajectory_transformed = svd.fit_transform(trajectory)
            trajectory_reconstructed = trajectory_transformed @ svd.components_
            
            # Восстанавливаем исходный ряд с помощью диагонального усреднения
            reconstructed = np.zeros(X.shape[0])
            counts = np.zeros(X.shape[0])
            
            for j in range(K):
                for l in range(window_size):
                    idx = j + l
                    reconstructed[idx] += trajectory_reconstructed[j, l]
                    counts[idx] += 1
            
            # Нормализуем суммой количества элементов
            reconstructed /= np.maximum(counts, 1)
            
            # Обновляем признак
            X_smoothed[:, i] = reconstructed
        
        return X_smoothed
    
    def preprocess_impulse_noise(self, X):
        """Улучшенная предобработка данных с импульсным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        X_processed = X.copy()
        
        # Используем комбинацию методов для обнаружения и обработки выбросов
        
        # 1. Метод DBSCAN для выявления выбросов
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            
            # Обрабатываем каждый признак отдельно
            for i in range(X_processed.shape[1]):
                feature = X_processed[:, i].reshape(-1, 1)
                
                # Автоматически определяем параметр eps для DBSCAN
                # на основе расстояний до K ближайших соседей
                k = min(5, len(feature) - 1)
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(feature)
                distances, _ = knn.kneighbors(feature)
                distances = np.sort(distances[:, -1])
                
                # Находим "колено" в графике расстояний для определения eps
                knee_point = self._find_knee_point(distances)
                eps = distances[knee_point] if knee_point > 0 else np.mean(distances)
                
                # Определяем min_samples на основе размера данных
                min_samples = max(2, min(int(len(feature) * 0.05), 5))
                
                # Применяем DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(feature)
                
                # Находим выбросы (помеченные как -1)
                outlier_mask = labels == -1
                
                # Если найдены выбросы, заменяем их интерполированными значениями
                if np.any(outlier_mask):
                    # Вычисляем медиану по невыбросам
                    inlier_median = np.median(feature[~outlier_mask])
                    
                    # Заменяем выбросы медианными значениями
                    X_processed[outlier_mask, i] = inlier_median
        except:
            # Если DBSCAN не сработал, используем более простой метод на основе z-score
            self._handle_outliers_zscore(X_processed)
        
        # 2. Применяем адаптивный медианный фильтр для финальной обработки
        try:
            from scipy.ndimage import median_filter
            
            # Обрабатываем каждый признак отдельно
            for i in range(X_processed.shape[1]):
                feature = X_processed[:, i].flatten()
                
                # Применяем медианный фильтр с адаптивным размером окна
                # Более высокая дисперсия данных -> большее окно
                std_value = np.std(feature)
                window_size = 3
                if std_value > 1.0:
                    window_size = 5
                if std_value > 2.0:
                    window_size = 7
                
                # Убедимся, что размер окна не превышает размер данных
                window_size = min(window_size, len(feature) - 1)
                if window_size > 1:
                    filtered = median_filter(feature, size=window_size)
                    X_processed[:, i] = filtered
        except:
            # Если медианный фильтр не сработал, оставляем как есть
            pass
        
        return X_processed
    
    def _find_knee_point(self, distances):
        """Находит точку колена в графике расстояний для определения параметра eps
        
        Args:
            distances: Отсортированный массив расстояний
            
        Returns:
            knee_idx: Индекс точки колена
        """
        # Нормализуем данные
        x = np.arange(len(distances)) / len(distances)
        y = distances / np.max(distances)
        
        # Находим точку, наиболее удаленную от прямой, соединяющей первую и последнюю точки
        # (метод "maximum distance")
        a = np.array([0, y[0]])
        b = np.array([1, y[-1]])
        
        max_dist = 0
        knee_idx = 0
        
        for i in range(len(y)):
            p = np.array([x[i], y[i]])
            dist = np.abs(np.cross(b-a, a-p)) / np.linalg.norm(b-a)
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return knee_idx
    
    def _handle_outliers_zscore(self, X):
        """Обрабатывает выбросы с использованием z-score метода
        
        Args:
            X: Данные для обработки (изменяются на месте)
        """
        # Вычисляем z-оценки
        z_scores = stats.zscore(X, axis=0, nan_policy='omit')
        
        # Находим выбросы (|z| > 3)
        abs_z_scores = np.abs(z_scores)
        outlier_mask = abs_z_scores > 3
        
        # Вычисляем медианы для каждого признака
        medians = np.nanmedian(X, axis=0)
        
        # Заменяем выбросы
        for i in range(X.shape[1]):
            column_outliers = outlier_mask[:, i]
            X[column_outliers, i] = medians[i]
    
    def preprocess_missing_values(self, X):
        """Улучшенная предобработка данных с пропущенными значениями
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        X_processed = X.copy()
        
        # Определяем процент пропущенных значений в каждом столбце
        missing_percent = np.sum(np.isnan(X_processed), axis=0) / X_processed.shape[0]
        
        # 1. Для столбцов с небольшим количеством пропусков (< 15%)
        # используем KNN-импутацию
        low_missing_cols = missing_percent < 0.15
        if np.any(low_missing_cols):
            try:
                from sklearn.impute import KNNImputer
                
                # Определяем оптимальное количество соседей
                n_neighbors = min(5, max(1, X_processed.shape[0] // 10))
                
                # Применяем KNN-импутацию только для столбцов с небольшим количеством пропусков
                X_low_missing = X_processed[:, low_missing_cols]
                
                # Проверяем, что у нас достаточно данных для импутации
                if X_low_missing.shape[0] > n_neighbors:
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    X_imputed = imputer.fit_transform(X_low_missing)
                    X_processed[:, low_missing_cols] = X_imputed
            except:
                # Если KNN не сработал, используем медианную импутацию
                for i in np.where(low_missing_cols)[0]:
                    col_data = X_processed[:, i]
                    mask = np.isnan(col_data)
                    if np.any(mask):
                        median_val = np.nanmedian(col_data)
                        X_processed[mask, i] = median_val
        
        # 2. Для столбцов с большим количеством пропусков (>= 15%)
        # используем MICE (Multiple Imputation by Chained Equations)
        high_missing_cols = ~low_missing_cols
        if np.any(high_missing_cols):
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                from sklearn.ensemble import ExtraTreesRegressor
                
                # Используем ExtraTrees для итеративной импутации
                estimator = ExtraTreesRegressor(n_estimators=50, random_state=42)
                
                # Настраиваем MICE с максимальным количеством итераций
                mice_imputer = IterativeImputer(
                    estimator=estimator,
                    max_iter=10,
                    initial_strategy='median',
                    random_state=42
                )
                
                # Сначала заполняем все пропуски медианными значениями
                for i in range(X_processed.shape[1]):
                    col_data = X_processed[:, i]
                    mask = np.isnan(col_data)
                    if np.any(mask):
                        median_val = np.nanmedian(col_data)
                        X_processed[mask, i] = median_val
                
                # Затем применяем MICE
                X_processed = mice_imputer.fit_transform(X_processed)
            except:
                # Если MICE не сработал, используем медианную импутацию для всех столбцов
                for i in range(X_processed.shape[1]):
                    col_data = X_processed[:, i]
                    mask = np.isnan(col_data)
                    if np.any(mask):
                        median_val = np.nanmedian(col_data)
                        X_processed[mask, i] = median_val
        
        # 3. Финальная проверка на оставшиеся NaN
        if np.any(np.isnan(X_processed)):
            # Заполняем оставшиеся NaN медианами
            for i in range(X_processed.shape[1]):
                col_data = X_processed[:, i]
                mask = np.isnan(col_data)
                if np.any(mask):
                    median_val = np.nanmedian(col_data)
                    if np.isnan(median_val):  # Если вся колонка NaN
                        median_val = 0
                    X_processed[mask, i] = median_val
        
        return X_processed
    
    def preprocess_uniform_noise(self, X):
        """Улучшенная предобработка данных с равномерным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для равномерного шума используем комбинацию методов:
        # 1. Вейвлет-фильтрация для удаления высокочастотного шума
        # 2. Локально-взвешенное сглаживание для сохранения структуры данных
        
        X_processed = X.copy()
        
        # Пробуем применить вейвлет-фильтрацию
        try:
            import pywt
            
            # Обрабатываем каждый признак отдельно
            for i in range(X_processed.shape[1]):
                feature = X_processed[:, i].flatten()
                
                # Для вейвлет-преобразования требуется длина, кратная степени 2
                # Находим ближайшую степень 2, большую или равную длине данных
                n = len(feature)
                power = 2 ** np.ceil(np.log2(n)).astype(int)
                
                # Дополняем данные до нужной длины
                padded = np.pad(feature, (0, power - n), 'symmetric')
                
                # Выбираем вейвлет в зависимости от характеристик данных
                wavelet = 'db4'  # Daubechies 4 - хороший выбор для большинства данных
                
                # Выполняем декомпозицию
                coeffs = pywt.wavedec(padded, wavelet, level=4)
                
                # Пороговая обработка вейвлет-коэффициентов
                # (удаляем шум, сохраняя структуру)
                sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
                threshold = sigma * np.sqrt(2 * np.log(len(padded)))
                
                # Применяем мягкую пороговую обработку
                new_coeffs = []
                for j, coeff in enumerate(coeffs):
                    if j == 0:  # Аппроксимирующие коэффициенты не изменяем
                        new_coeffs.append(coeff)
                    else:  # Детализирующие коэффициенты обрабатываем
                        new_coeff = pywt.threshold(coeff, threshold, 'soft')
                        new_coeffs.append(new_coeff)
                
                # Восстанавливаем сигнал
                reconstructed = pywt.waverec(new_coeffs, wavelet)
                
                # Возвращаем оригинальную длину
                X_processed[:, i] = reconstructed[:n]
        except:
            # Если вейвлет-фильтрация не сработала, используем локально-взвешенное сглаживание
            self._apply_lowess(X_processed)
        
        return X_processed
    
    def _apply_lowess(self, X, frac=0.3):
        """Применяет локально-взвешенное сглаживание (LOWESS)
        
        Args:
            X: Данные для обработки (изменяются на месте)
            frac: Параметр сглаживания (доля точек, используемых для локальной регрессии)
        """
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Обрабатываем каждый признак отдельно
            for i in range(X.shape[1]):
                feature = X[:, i].flatten()
                
                # Создаем индексы (ось x)
                x = np.arange(len(feature))
                
                # Применяем LOWESS
                smoothed = lowess(
                    feature, x,
                    frac=frac,
                    it=2,  # Количество итераций повторного взвешивания
                    return_sorted=False
                )
                
                # Обновляем значения
                X[:, i] = smoothed
        except:
            # Если LOWESS не сработал, используем простое скользящее среднее
            window_size = max(3, len(X) // 10)
            window_size = min(window_size, len(X) - 1)  # Не больше размера данных
            
            if window_size > 1:
                for i in range(X.shape[1]):
                    feature = X[:, i].flatten()
                    
                    # Применяем скользящее среднее
                    smoothed = np.convolve(feature, np.ones(window_size)/window_size, mode='same')
                    
                    # Исправляем края (где свертка дает искаженные результаты)
                    half_window = window_size // 2
                    smoothed[:half_window] = feature[:half_window]
                    smoothed[-half_window:] = feature[-half_window:]
                    
                    # Обновляем значения
                    X[:, i] = smoothed
    
    def preprocess_salt_pepper_noise(self, X):
        """Улучшенная предобработка данных с шумом типа "соль и перец"
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для шума "соль и перец" оптимален адаптивный медианный фильтр
        # с обнаружением экстремальных значений
        
        X_processed = X.copy()
        
        # Находим глобальные минимумы и максимумы для каждого признака
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        # Вычисляем предполагаемые пороги для "соли" и "перца"
        percentiles = np.percentile(X, [1, 99], axis=0)
        lower_bounds = percentiles[0]
        upper_bounds = percentiles[1]
        
        # Обнаруживаем и заменяем экстремальные значения
        for i in range(X.shape[1]):
            feature = X[:, i]
            
            # Находим значения, близкие к экстремумам (потенциальные "соль" и "перец")
            salt_mask = np.isclose(feature, max_vals[i], rtol=1e-2) | (feature > upper_bounds[i])
            pepper_mask = np.isclose(feature, min_vals[i], rtol=1e-2) | (feature < lower_bounds[i])
            
            # Если обнаружены экстремумы, заменяем их медианными значениями из соседей
            if np.any(salt_mask) or np.any(pepper_mask):
                noise_mask = salt_mask | pepper_mask
                
                # Получаем индексы шумовых точек
                noise_indices = np.where(noise_mask)[0]
                
                for idx in noise_indices:
                    # Определяем локальное окно (соседние точки)
                    window_start = max(0, idx - 2)
                    window_end = min(len(feature), idx + 3)
                    
                    # Получаем значения из окна, исключая шумовые точки
                    window_values = feature[window_start:window_end].copy()
                    window_noise_mask = noise_mask[window_start:window_end]
                    clean_values = window_values[~window_noise_mask]
                    
                    # Если в окне есть чистые значения, используем их медиану
                    if len(clean_values) > 0:
                        X_processed[idx, i] = np.median(clean_values)
                    else:
                        # Если все значения в окне - шум, используем глобальную медиану
                        X_processed[idx, i] = np.median(feature[~noise_mask])
        
        # Дополнительно применяем адаптивный медианный фильтр
        try:
            from scipy.ndimage import median_filter
            
            # Проверяем, сколько точек было заменено
            replaced_ratio = np.mean(np.any(X != X_processed, axis=1))
            
            # Если заменено более 5% точек, применяем дополнительную фильтрацию
            if replaced_ratio > 0.05:
                for i in range(X.shape[1]):
                    feature = X_processed[:, i].flatten()
                    
                    # Выбираем размер окна в зависимости от размера данных
                    window_size = min(5, max(3, len(feature) // 20))
                    
                    # Применяем медианный фильтр
                    filtered = median_filter(feature, size=window_size)
                    X_processed[:, i] = filtered
        except:
            # Если медианный фильтр не сработал, оставляем как есть
            pass
        
        return X_processed
    
    def preprocess_multiplicative_noise(self, X):
        """Улучшенная предобработка данных с мультипликативным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для мультипликативного шума оптимально:
        # 1. Логарифмическое преобразование (превращает мультипликативный шум в аддитивный)
        # 2. Обработка аддитивного шума
        # 3. Обратное экспоненциальное преобразование
        
        X_processed = X.copy()
        
        # Проверяем на отрицательные значения
        has_negative = np.any(X <= 0)
        
        # 1. Логарифмическое преобразование
        if has_negative:
            # Если есть отрицательные значения, сдвигаем данные в положительную область
            min_vals = np.min(X, axis=0)
            shift = np.where(min_vals <= 0, np.abs(min_vals) + 1, 0)
            X_log = np.log1p(X + shift.reshape(1, -1))
        else:
            # Если все значения положительные, используем стандартное логарифмирование
            X_log = np.log1p(np.abs(X))
        
        # 2. Обработка аддитивного шума (применяем методы для гауссовского шума)
        X_log_processed = self.preprocess_gaussian_noise(X_log)
        
        # 3. Обратное экспоненциальное преобразование
        if has_negative:
            X_processed = np.expm1(X_log_processed) - shift.reshape(1, -1)
        else:
            X_processed = np.expm1(X_log_processed)
        
        # Дополнительно применяем адаптивное сглаживание
        feature_std = np.std(X_processed, axis=0)
        
        for i in range(X.shape[1]):
            if feature_std[i] > 1.0:
                # Для признаков с высокой дисперсией применяем TVD (Total Variation Denoising)
                try:
                    from skimage.restoration import denoise_tv_chambolle
                    
                    feature = X_processed[:, i].reshape(-1)
                    # Нормализуем данные для TVD
                    feature_min = np.min(feature)
                    feature_max = np.max(feature)
                    feature_norm = (feature - feature_min) / (feature_max - feature_min + 1e-10)
                    
                    # Применяем TVD с адаптивным параметром регуляризации
                    weight = 0.1
                    denoised = denoise_tv_chambolle(feature_norm, weight=weight)
                    
                    # Возвращаем к исходному масштабу
                    denoised = denoised * (feature_max - feature_min) + feature_min
                    X_processed[:, i] = denoised
                except:
                    # Если TVD не сработал, используем простое сглаживание
                    window_size = min(5, len(X_processed) - 1)
                    if window_size > 1:
                        feature = X_processed[:, i].reshape(-1)
                        smoothed = np.convolve(feature, np.ones(window_size)/window_size, mode='same')
                        X_processed[:, i] = smoothed
        
        return X_processed
    
    def preprocess_data(self, X, noise_type):
        """Предобрабатывает данные в зависимости от типа шума
        
        Args:
            X: Зашумленные данные
            noise_type: Тип шума
            
        Returns:
            X_processed: Обработанные данные
        """
        # Сначала проверяем на пропущенные значения (NaN) независимо от типа шума
        if np.any(np.isnan(X)):
            X = self.preprocess_missing_values(X)
        
        # Затем применяем соответствующую предобработку по типу шума
        preprocessing_methods = {
            'gaussian': self.preprocess_gaussian_noise,
            'uniform': self.preprocess_uniform_noise,
            'impulse': self.preprocess_impulse_noise,
            'missing': self.preprocess_missing_values,
            'salt_pepper': self.preprocess_salt_pepper_noise,
            'multiplicative': self.preprocess_multiplicative_noise
        }
        
        if noise_type in preprocessing_methods:
            return preprocessing_methods[noise_type](X)
        else:
            print(f"Предупреждение: Предобработка для шума типа '{noise_type}' не реализована.")
            return X

    def detect_noise_type(self, X_clean, X_noisy):
        """Определяет тип шума в данных
        
        Args:
            X_clean: Чистые данные (если доступны)
            X_noisy: Зашумленные данные
            
        Returns:
            noise_type: Предполагаемый тип шума
            confidence: Уверенность в определении (0-1)
        """
        # Если чистые данные недоступны, невозможно определить тип шума
        if X_clean is None:
            return None, 0.0
        
        # Вычисляем разницу между чистыми и зашумленными данными
        diff = X_noisy - X_clean
        
        # Извлекаем статистики для определения типа шума
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        diff_skew = stats.skew(diff.flatten())
        diff_kurtosis = stats.kurtosis(diff.flatten())
        
        # Вычисляем корреляцию между абсолютной разницей и значениями чистых данных
        abs_diff = np.abs(diff)
        correlation = np.corrcoef(abs_diff.flatten(), np.abs(X_clean).flatten())[0, 1]
        
        # Проверяем на пропущенные значения
        missing_ratio = np.mean(np.isnan(X_noisy))
        
        # Находим экстремальные значения
        percentiles = np.percentile(X_clean, [1, 99])
        extremes_ratio = np.mean((X_noisy < percentiles[0]) | (X_noisy > percentiles[1]))
        
        # Оцениваем вероятности для каждого типа шума
        
        # Гауссовский шум: симметричное распределение с нулевым средним
        gaussian_score = 1.0 - min(1.0, abs(diff_skew) / 0.5 + abs(diff_kurtosis - 3) / 5)
        
        # Равномерный шум: равномерное распределение, низкий эксцесс
        uniform_score = 1.0 - min(1.0, abs(diff_kurtosis) / 3)
        
        # Импульсный шум: высокий эксцесс, много выбросов
        impulse_score = min(1.0, abs(diff_kurtosis) / 10)
        
        # Шум "соль и перец": высокая доля экстремальных значений
        salt_pepper_score = min(1.0, extremes_ratio * 5)
        
        # Мультипликативный шум: высокая корреляция между шумом и сигналом
        multiplicative_score = min(1.0, abs(correlation) * 2)
        
        # Пропущенные значения: наличие NaN
        missing_score = min(1.0, missing_ratio * 10)
        
        # Определяем наиболее вероятный тип шума
        scores = {
            'gaussian': gaussian_score,
            'uniform': uniform_score,
            'impulse': impulse_score,
            'salt_pepper': salt_pepper_score,
            'multiplicative': multiplicative_score,
            'missing': missing_score
        }
        
        max_score_type = max(scores.items(), key=lambda x: x[1])
        
        return max_score_type[0], max_score_type[1]
    
    def advanced_noise_reduction(self, X):
        """Продвинутый метод шумоподавления на основе вейвлет-преобразования
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Данные с удаленным шумом
        """
        try:
            import pywt
            
            X_processed = X.copy()
            
            # Обрабатываем каждый признак отдельно
            for i in range(X.shape[1]):
                feature = X[:, i].flatten()
                
                # Пропускаем признаки с малым количеством элементов
                if len(feature) < 16:  # Минимальный размер для вейвлет-преобразования
                    continue
                    
                # Подбираем подходящий размер для вейвлет-преобразования (степень 2)
                n = len(feature)
                pad_len = int(2**np.ceil(np.log2(n)))
                padded = np.pad(feature, (0, pad_len - n), 'symmetric')
                
                # Выбираем вейвлет-преобразование в зависимости от характеристик данных
                # db4 (Daubechies 4) хорошо подходит для большинства сигналов
                wavelet = 'db4'
                
                # Разложение с многоуровневой декомпозицией
                max_level = pywt.dwt_max_level(len(padded), wavelet)
                level = min(4, max_level)  # Используем не более 4 уровней
                
                # Выполняем декомпозицию
                coeffs = pywt.wavedec(padded, wavelet, level=level)
                
                # Адаптивная пороговая обработка коэффициентов для удаления шума
                # Используем универсальный порог VisuShrink
                sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
                
                # Модифицированный порог для каждого уровня декомпозиции
                # Более высокий для высокочастотных компонентов
                new_coeffs = [coeffs[0]]  # Аппроксимация (не изменяется)
                
                for j in range(1, len(coeffs)):
                    # Адаптивный порог зависит от уровня декомпозиции
                    # Более низкие уровни = высокие частоты = больше шума
                    level_sigma = sigma * (1.0 / (j + 1))**0.5
                    threshold = level_sigma * np.sqrt(2 * np.log(len(padded)))
                    
                    # Мягкая пороговая обработка для сохранения важных деталей
                    new_coeffs.append(pywt.threshold(coeffs[j], threshold, 'soft'))
                
                # Реконструкция сигнала
                denoised = pywt.waverec(new_coeffs, wavelet)
                
                # Обрезаем до исходной длины и обновляем данные
                X_processed[:, i] = denoised[:n]
            
            return X_processed
        except ImportError:
            print("Предупреждение: pywt не установлен, используется стандартное шумоподавление")
            return X

class ModelBuilder:
    """Класс для построения и оптимизации моделей классификации"""
    
    def __init__(self):
        """Инициализирует построитель моделей"""
        self.models = {}
        self.best_params = {}
        self.feature_scaler = RobustScaler()  # Более устойчив к выбросам
        self.feature_selector = None
        self.pca = None

    def build_main_neural_network(self, input_shape, num_classes, hyperparams=None):
        """Строит улучшенную нейронную сеть с заданными гиперпараметрами и резидуальными соединениями
        
        Args:
            input_shape: Размерность входных данных
            num_classes: Количество классов
            hyperparams: Словарь с гиперпараметрами (если None, используются значения по умолчанию)
            
        Returns:
            model: Скомпилированная модель нейронной сети
        """
        if hyperparams is None:
            # Значения по умолчанию с расширенными опциями
            hyperparams = {
                'units_1': 256,
                'units_2': 128,
                'units_3': 64,
                'units_4': 32,
                'dropout_rate': 0.35,
                'learning_rate': 0.0015,
                'l2_reg': 0.0015,
                'batch_size': 64,
                'activation': 'swish',
                'leaky_alpha': 0.2,
                'noise_stddev': 0.15,
                'use_bn': True,
                'use_residual': True,
                'residual_scaling': 0.15
            }
        
        # Определяем пользовательские функции активации
        def swish(x):
            return x * tf.nn.sigmoid(x)
        
        def mish(x):
            return x * tf.nn.tanh(tf.nn.softplus(x))
        
        # Создаем модель с улучшенной архитектурой
        inputs = Input(shape=input_shape)
        
        # Добавляем слой шума для повышения устойчивости
        x = GaussianNoise(hyperparams.get('noise_stddev', 0.15))(inputs)
        
        # Первый блок с выбором функции активации
        x1 = Dense(hyperparams['units_1'], 
                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']),
                kernel_initializer='he_normal')(x)
        if hyperparams.get('use_bn', True):
            x1 = BatchNormalization()(x1)
        
        # Применяем выбранную функцию активации
        activation_type = hyperparams.get('activation', 'swish')
        if activation_type == 'leaky_relu':
            x1 = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x1)
        elif activation_type == 'swish':
            x1 = layers.Lambda(swish)(x1)
        elif activation_type == 'mish':
            x1 = layers.Lambda(mish)(x1)
        else:
            x1 = Activation(activation_type)(x1)
        
        # Применяем стандартный Dropout вместо SpatialDropout1D
        x1 = Dropout(hyperparams['dropout_rate'])(x1)
        
        # Второй блок с резидуальным соединением
        x2 = Dense(hyperparams['units_2'], 
                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.9),
                kernel_initializer='he_normal')(x1)
        if hyperparams.get('use_bn', True):
            x2 = BatchNormalization()(x2)
        
        # Применяем выбранную функцию активации
        if activation_type == 'leaky_relu':
            x2 = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x2)
        elif activation_type == 'swish':
            x2 = layers.Lambda(swish)(x2)
        elif activation_type == 'mish':
            x2 = layers.Lambda(mish)(x2)
        else:
            x2 = Activation(activation_type)(x2)
        
        # Добавляем резидуальное соединение, если включено
        if hyperparams.get('use_residual', True):
            # Проекция первого слоя для соответствия размерности
            if hyperparams['units_1'] != hyperparams['units_2']:
                x1_proj = Dense(hyperparams['units_2'], 
                                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.7), 
                                kernel_initializer='he_normal',
                                use_bias=False)(x1)
            else:
                x1_proj = x1
            
            # Масштабируем и добавляем резидуальное соединение
            residual_scale = hyperparams.get('residual_scaling', 0.15)
            x2 = layers.add([x2, x1_proj * residual_scale])
        
        x2 = Dropout(hyperparams['dropout_rate'] * 0.8)(x2)
        
        # Третий блок с резидуальным соединением
        x3 = Dense(hyperparams['units_3'], 
                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.8),
                kernel_initializer='he_normal')(x2)
        if hyperparams.get('use_bn', True):
            x3 = BatchNormalization()(x3)
        
        # Применяем выбранную функцию активации
        if activation_type == 'leaky_relu':
            x3 = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x3)
        elif activation_type == 'swish':
            x3 = layers.Lambda(swish)(x3)
        elif activation_type == 'mish':
            x3 = layers.Lambda(mish)(x3)
        else:
            x3 = Activation(activation_type)(x3)
        
        # Добавляем резидуальное соединение, если включено
        if hyperparams.get('use_residual', True):
            # Проекция второго слоя для соответствия размерности
            if hyperparams['units_2'] != hyperparams['units_3']:
                x2_proj = Dense(hyperparams['units_3'], 
                                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.7), 
                                kernel_initializer='he_normal',
                                use_bias=False)(x2)
            else:
                x2_proj = x2
            
            # Масштабируем и добавляем резидуальное соединение
            residual_scale = hyperparams.get('residual_scaling', 0.15)
            x3 = layers.add([x3, x2_proj * residual_scale])
        
        x3 = Dropout(hyperparams['dropout_rate'] * 0.6)(x3)
        
        # Четвертый блок с резидуальным соединением
        x4 = Dense(hyperparams['units_4'], 
                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.7),
                kernel_initializer='he_normal')(x3)
        if hyperparams.get('use_bn', True):
            x4 = BatchNormalization()(x4)
        
        # Применяем выбранную функцию активации
        if activation_type == 'leaky_relu':
            x4 = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x4)
        elif activation_type == 'swish':
            x4 = layers.Lambda(swish)(x4)
        elif activation_type == 'mish':
            x4 = layers.Lambda(mish)(x4)
        else:
            x4 = Activation(activation_type)(x4)
        
        # Добавляем резидуальное соединение, если включено
        if hyperparams.get('use_residual', True):
            # Проекция третьего слоя для соответствия размерности
            if hyperparams['units_3'] != hyperparams['units_4']:
                x3_proj = Dense(hyperparams['units_4'], 
                                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.6), 
                                kernel_initializer='he_normal',
                                use_bias=False)(x3)
            else:
                x3_proj = x3
            
            # Масштабируем и добавляем резидуальное соединение
            residual_scale = hyperparams.get('residual_scaling', 0.15)
            x4 = layers.add([x4, x3_proj * residual_scale])
        
        x4 = Dropout(hyperparams['dropout_rate'] * 0.4)(x4)
        
        # Добавляем дополнительный выходной слой для более глубокого представления
        x5 = Dense(hyperparams['units_4'] // 2, 
                kernel_regularizer=regularizers.l2(hyperparams['l2_reg']*0.5),
                kernel_initializer='he_normal')(x4)
        x5 = BatchNormalization()(x5)
        
        # Применяем выбранную функцию активации
        if activation_type == 'leaky_relu':
            x5 = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x5)
        elif activation_type == 'swish':
            x5 = layers.Lambda(swish)(x5)
        elif activation_type == 'mish':
            x5 = layers.Lambda(mish)(x5)
        else:
            x5 = Activation(activation_type)(x5)
        
        # Выходной слой с соответствующей функцией активации
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid', 
                        kernel_initializer='glorot_uniform')(x5)
        else:
            outputs = Dense(num_classes, activation='softmax', 
                        kernel_initializer='glorot_uniform')(x5)
        
        # Создаем модель
        model = Model(inputs=inputs, outputs=outputs)
        
        # Создаем оптимизатор с опциональным clipnorm для стабильности
        optimizer = Adam(
            learning_rate=hyperparams['learning_rate'],
            clipnorm=hyperparams.get('clipnorm', 1.0),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Выбираем функцию потерь в зависимости от типа задачи
        loss_type = hyperparams.get('loss_type', 'focal')  # По умолчанию используем Focal Loss
        
        if num_classes == 2:
            if loss_type == 'default':
                loss = 'binary_crossentropy'
            elif loss_type == 'focal':
                # Используем глобально определенный класс для Focal Loss
                gamma = hyperparams.get('focal_gamma', 2.0)
                alpha = hyperparams.get('focal_alpha', 0.25)
                loss = FocalLoss(gamma=gamma, alpha=alpha)
            else:
                loss = 'binary_crossentropy'
                
            metrics = ['accuracy']
        else:
            if loss_type == 'default':
                loss = 'categorical_crossentropy'
            elif loss_type == 'focal':
                # Используем глобально определенный класс для Focal Loss
                gamma = hyperparams.get('focal_gamma', 2.0)
                loss = CategoricalFocalLoss(gamma=gamma)
            else:
                loss = 'categorical_crossentropy'
                
            metrics = ['accuracy']
        
        # Компилируем модель
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def optimize_neural_network(self, X_train, y_train, X_val, y_val, input_shape, num_classes, n_trials=50, noise_type=None):
        """Оптимизирует гиперпараметры нейронной сети с помощью Optuna с учетом типа шума
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            input_shape: Размерность входных данных
            num_classes: Количество классов
            n_trials: Количество испытаний оптимизации
            noise_type: Тип шума для специализированной оптимизации
            
        Returns:
            best_params: Лучшие найденные гиперпараметры
        """
        import optuna
        from optuna.samplers import TPESampler
        
        # Улучшенные диапазоны параметров в зависимости от типа шума
        if noise_type == 'gaussian':
            # Для гауссовского шума важна сильная регуляризация и сглаживание градиентов
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
                    'learning_rate': trial.suggest_float('learning_rate', 5e-5, 3e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 5e-4, 5e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'relu', 'elu']),
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.15, 0.3),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.1, 0.25),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 3.0),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.3) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.8, 1.2)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        elif noise_type == 'impulse':
            # Для импульсного шума важна устойчивость к выбросам
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.35, 0.6),
                    'learning_rate': trial.suggest_float('learning_rate', 2e-5, 1e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'leaky_relu', 'mish']),
                    'leaky_alpha': trial.suggest_float('leaky_alpha', 0.1, 0.3) if trial.suggest_categorical('activation', ['swish', 'leaky_relu', 'mish']) == 'leaky_relu' else 0.2,
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.2, 0.4),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.15, 0.3),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 2.0, 4.0),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.25, 0.4) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.5, 1.0)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        elif noise_type == 'missing':
            # Для пропущенных значений
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.4, 0.6),
                    'learning_rate': trial.suggest_float('learning_rate', 5e-5, 2e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'mish', 'relu']),
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.2, 0.4),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.1, 0.25),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 3.0),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.4) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.8, 1.2)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        elif noise_type == 'salt_pepper':
            # Для шума типа "соль и перец"
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.4, 0.65),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-4, 5e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'activation': trial.suggest_categorical('activation', ['leaky_relu', 'swish', 'mish']),
                    'leaky_alpha': trial.suggest_float('leaky_alpha', 0.1, 0.3) if trial.suggest_categorical('activation', ['leaky_relu', 'swish', 'mish']) == 'leaky_relu' else 0.2,
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.2, 0.5),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.15, 0.35),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 2.0, 4.0),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.4) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.5, 1.0)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        elif noise_type == 'multiplicative':
            # Для мультипликативного шума
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.55),
                    'learning_rate': trial.suggest_float('learning_rate', 5e-5, 2e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'relu', 'elu']),
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.1, 0.3),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.1, 0.2),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 3.0),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.35) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.7, 1.2)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        elif noise_type == 'uniform':
            # Для равномерного шума
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.25, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'relu', 'elu']),
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.1, 0.3),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.1, 0.2),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 2.5),
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.35) if num_classes == 2 else None,
                    'clipnorm': trial.suggest_float('clipnorm', 0.8, 1.2)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        else:
            # Для неизвестного типа шума или общего случая
            def objective(trial):
                hyperparams = {
                    'units_1': trial.suggest_int('units_1', 128, 512, step=32),
                    'units_2': trial.suggest_int('units_2', 64, 256, step=32),
                    'units_3': trial.suggest_int('units_3', 32, 128, step=16),
                    'units_4': trial.suggest_int('units_4', 16, 64, step=8),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['swish', 'relu', 'leaky_relu', 'elu', 'mish']),
                    'leaky_alpha': trial.suggest_float('leaky_alpha', 0.05, 0.3) if trial.suggest_categorical('activation', ['swish', 'relu', 'leaky_relu', 'elu', 'mish']) == 'leaky_relu' else 0.2,
                    'noise_stddev': trial.suggest_float('noise_stddev', 0.05, 0.3),
                    'use_bn': trial.suggest_categorical('use_bn', [True]),
                    'use_residual': trial.suggest_categorical('use_residual', [True]),
                    'residual_scaling': trial.suggest_float('residual_scaling', 0.05, 0.3),
                    'loss_type': trial.suggest_categorical('loss_type', ['focal', 'default']),
                    'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0) if trial.suggest_categorical('loss_type', ['focal', 'default']) == 'focal' else 2.0,
                    'focal_alpha': trial.suggest_float('focal_alpha', 0.2, 0.4) if num_classes == 2 and trial.suggest_categorical('loss_type', ['focal', 'default']) == 'focal' else 0.25,
                    'clipnorm': trial.suggest_float('clipnorm', 0.5, 1.5)
                }
                return self._evaluate_neural_network(hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes)
        
        # Создаем исследование Optuna с более эффективным сэмплером
        # Используем TPESampler с multivariate=True для лучшего поиска в высокомерном пространстве
        sampler = TPESampler(multivariate=True, seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Используем обратные вызовы для более эффективного поиска
        def print_callback(study, trial):
            if trial.number % 5 == 0:
                print(f"Завершен поиск {trial.number}/{n_trials}. Текущее лучшее значение: {study.best_value:.4f}")
        
        # Оптимизируем с большим числом попыток
        study.optimize(objective, n_trials=n_trials, callbacks=[print_callback])
        
        print("Оптимизация нейронной сети завершена:")
        print(f"Лучшие гиперпараметры: {study.best_params}")
        print(f"Лучшее значение целевой функции: {study.best_value:.4f}")
        
        # Получаем лучшие параметры
        best_params = study.best_params
        
        # Добавляем параметры, которые могли быть не определены
        if 'focal_gamma' not in best_params and best_params.get('loss_type') == 'focal':
            best_params['focal_gamma'] = 2.0
        
        if 'focal_alpha' not in best_params and best_params.get('loss_type') == 'focal' and num_classes == 2:
            best_params['focal_alpha'] = 0.25
        
        if 'leaky_alpha' not in best_params and best_params.get('activation') == 'leaky_relu':
            best_params['leaky_alpha'] = 0.2
        
        # Преобразуем результаты Optuna в полный словарь гиперпараметров
        best_hyperparams = {
            'units_1': best_params.get('units_1', 256),
            'units_2': best_params.get('units_2', 128),
            'units_3': best_params.get('units_3', 64),
            'units_4': best_params.get('units_4', 32),
            'dropout_rate': best_params.get('dropout_rate', 0.4),
            'learning_rate': best_params.get('learning_rate', 0.001),
            'l2_reg': best_params.get('l2_reg', 0.001),
            'batch_size': best_params.get('batch_size', 64),
            'activation': best_params.get('activation', 'swish'),
            'leaky_alpha': best_params.get('leaky_alpha', 0.2),
            'noise_stddev': best_params.get('noise_stddev', 0.15),
            'use_bn': best_params.get('use_bn', True),
            'use_residual': best_params.get('use_residual', True),
            'residual_scaling': best_params.get('residual_scaling', 0.15),
            'loss_type': best_params.get('loss_type', 'focal'),
            'focal_gamma': best_params.get('focal_gamma', 2.0),
            'focal_alpha': best_params.get('focal_alpha', 0.25),
            'clipnorm': best_params.get('clipnorm', 1.0)
        }
        
        return best_hyperparams

    def _evaluate_neural_network(self, hyperparams, X_train, y_train, X_val, y_val, input_shape, num_classes):
        """Вспомогательный метод для оценки нейронной сети с заданными гиперпараметрами
        
        Args:
            hyperparams: Словарь гиперпараметров
            X_train, y_train, X_val, y_val: Данные для обучения и валидации
            input_shape: Размерность входных данных
            num_classes: Количество классов
            
        Returns:
            val_loss: Значение функции потерь на валидационном наборе
        """
        # Подготовка данных
        if num_classes > 2:
            y_train_cat = to_categorical(y_train)
            y_val_cat = to_categorical(y_val)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
        
        # Строим модель с текущими гиперпараметрами
        model = self.build_main_neural_network(input_shape, num_classes, hyperparams)
        
        # Ранняя остановка и уменьшение скорости обучения
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
        
        # Обучаем модель
        history = model.fit(
            X_train, y_train_cat,
            epochs=100,  # Увеличиваем количество эпох
            batch_size=hyperparams['batch_size'],
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Оцениваем модель
        val_loss = min(history.history['val_loss'])
        
        return val_loss
    
    def optimize_support_models(self, X_train, y_train, n_jobs=-1):
        """Оптимизирует гиперпараметры вспомогательных моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            n_jobs: Количество используемых процессов (-1 для использования всех)
            
        Returns:
            best_params: Словарь с лучшими параметрами для каждой модели
        """
        # Определяем расширенные пространства поиска для каждой модели
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'class_weight': [None, 'balanced']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'max_depth': [3, 5, 7, -1],
                'min_child_samples': [20, 30, 50]
            },
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        }
        
        # Модели для оптимизации
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'xgboost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'adaboost': AdaBoostClassifier(random_state=42)
        }
        
        best_params = {}
        
        # Оптимизируем каждую модель
        for name, model in base_models.items():
            print(f"\nОптимизация модели {name}...")
            
            # Создаем объект GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=5,  # Увеличиваем до 5-кратной кросс-валидации
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=0
            )
            
            # Обучаем на данных
            grid_search.fit(X_train, y_train)
            
            # Сохраняем лучшие параметры
            best_params[name] = grid_search.best_params_
            print(f"Лучшие параметры для {name}: {grid_search.best_params_}")
            print(f"Лучшая точность при кросс-валидации: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def perform_feature_selection(self, X_train, y_train, n_features=None):
        """Выполняет отбор признаков для улучшения качества моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            n_features: Количество признаков для отбора (если None, выбирается автоматически)
            
        Returns:
            X_train_selected: Преобразованные данные
        """
        if n_features is None:
            # Автоматически определяем оптимальное количество признаков (не менее 50%)
            n_features = max(int(X_train.shape[1] * 0.5), 2)
        
        # Используем ANOVA F-value для отбора признаков (для классификации)
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        print(f"Выполнен отбор признаков: из {X_train.shape[1]} оставлено {n_features}")
        
        # Можно применить PCA к отобранным признакам для дальнейшего улучшения
        # if X_train_selected.shape[1] > 10:
        #     self.pca = PCA(n_components=min(n_features, 10))
        #     X_train_selected = self.pca.fit_transform(X_train_selected)
        #     print(f"Применено PCA: финальная размерность {X_train_selected.shape[1]}")
        
        return X_train_selected
    
    def apply_feature_transformation(self, X):
        """Применяет преобразования признаков (отбор и PCA)
        
        Args:
            X: Исходные данные
            
        Returns:
            X_transformed: Преобразованные данные
        """
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        return X

    def build_ensemble_model(self, input_shape, num_classes, nn_params, support_params):
        """Строит расширенную ансамблевую модель с основной нейронной сетью и вспомогательными алгоритмами
        
        Args:
            input_shape: Размерность входных данных
            num_classes: Количество классов
            nn_params: Гиперпараметры нейронной сети
            support_params: Гиперпараметры вспомогательных моделей
            
        Returns:
            ensemble: Ансамблевая модель
        """
        # Создаем основную нейронную сеть
        main_nn = self.build_main_neural_network(input_shape, num_classes, nn_params)
        
        # Создаем вспомогательные модели с оптимизированными параметрами
        rf_params = support_params['random_forest']
        gb_params = support_params['gradient_boosting']
        svm_params = support_params['svm']
        knn_params = support_params['knn']
        xgb_params = support_params['xgboost']
        lgb_params = support_params['lightgbm']
        ada_params = support_params['adaboost']
        
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            bootstrap=rf_params.get('bootstrap', True),
            class_weight=rf_params.get('class_weight', None),
            random_state=42
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params['n_estimators'],
            learning_rate=gb_params['learning_rate'],
            max_depth=gb_params['max_depth'],
            min_samples_split=gb_params.get('min_samples_split', 2),
            subsample=gb_params.get('subsample', 1.0),
            random_state=42
        )
        
        svm_model = SVC(
            C=svm_params['C'],
            gamma=svm_params['gamma'],
            kernel=svm_params['kernel'],
            class_weight=svm_params.get('class_weight', None),
            probability=True,
            random_state=42
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=knn_params['n_neighbors'],
            weights=knn_params['weights'],
            p=knn_params['p'],
            algorithm=knn_params.get('algorithm', 'auto')
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth=xgb_params['max_depth'],
            min_child_weight=xgb_params.get('min_child_weight', 1),
            subsample=xgb_params.get('subsample', 1.0),
            colsample_bytree=xgb_params.get('colsample_bytree', 1.0),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=lgb_params['n_estimators'],
            learning_rate=lgb_params['learning_rate'],
            num_leaves=lgb_params.get('num_leaves', 31),
            max_depth=lgb_params['max_depth'],
            min_child_samples=lgb_params.get('min_child_samples', 20),
            random_state=42,
            verbose=-1
        )
        
        ada_model = AdaBoostClassifier(
            n_estimators=ada_params['n_estimators'],
            learning_rate=ada_params['learning_rate'],
            algorithm=ada_params.get('algorithm', 'SAMME.R'),
            random_state=42
        )
        
        # Также добавим дополнительную модель ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        
        # Сохраняем модели в словаре
        self.models = {
            'main_nn': main_nn,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'svm': svm_model,
            'knn': knn_model,
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'adaboost': ada_model,
            'extra_trees': et_model
        }
        
        self.best_params = {
            'nn_params': nn_params,
            'support_params': support_params
        }
        
        return self.models
    
    class ImprovedAdaptiveEnsemble:
        """Класс для улучшенного адаптивного ансамбля моделей"""
        
        def __init__(self, models, val_X=None, val_y=None, confidence_threshold=0.6): # confidence_threshold - уверенность NN
            """Инициализирует адаптивный ансамбль
            
            Args:
                models: Словарь с моделями
                val_X: Валидационные данные для калибровки весов
                val_y: Валидационные метки для калибровки весов
                confidence_threshold: Порог уверенности для основной модели
            """
            self.models = models
            self.confidence_threshold = confidence_threshold
            
            # Динамические веса для моделей
            self.model_weights = self._calculate_model_weights(val_X, val_y) if val_X is not None and val_y is not None else {
                'random_forest': 0.18,
                'gradient_boosting': 0.18,
                'svm': 0.12,
                'knn': 0.08,
                'xgboost': 0.18,
                'lightgbm': 0.18,
                'adaboost': 0.05,
                'extra_trees': 0.03
            }
            
            # Адаптивная схема весов для разных типов шума
            self.noise_type_weights = {
                'gaussian': {
                    'random_forest': 0.15,
                    'gradient_boosting': 0.15,
                    'svm': 0.10,
                    'knn': 0.05,
                    'xgboost': 0.20,
                    'lightgbm': 0.25,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                },
                'impulse': {
                    'random_forest': 0.20,
                    'gradient_boosting': 0.15,
                    'svm': 0.05,
                    'knn': 0.05,
                    'xgboost': 0.25,
                    'lightgbm': 0.20,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                },
                'missing': {
                    'random_forest': 0.15,
                    'gradient_boosting': 0.15,
                    'svm': 0.05,
                    'knn': 0.05,
                    'xgboost': 0.25,
                    'lightgbm': 0.25,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                },
                'salt_pepper': {
                    'random_forest': 0.20,
                    'gradient_boosting': 0.15,
                    'svm': 0.05,
                    'knn': 0.05,
                    'xgboost': 0.25,
                    'lightgbm': 0.20,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                },
                'multiplicative': {
                    'random_forest': 0.15,
                    'gradient_boosting': 0.15,
                    'svm': 0.10,
                    'knn': 0.05,
                    'xgboost': 0.25,
                    'lightgbm': 0.20,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                },
                'uniform': {
                    'random_forest': 0.15,
                    'gradient_boosting': 0.15,
                    'svm': 0.10,
                    'knn': 0.05,
                    'xgboost': 0.20,
                    'lightgbm': 0.25,
                    'adaboost': 0.05,
                    'extra_trees': 0.05
                }
            }
            
            print("Веса моделей в ансамбле:")
            for model_name, weight in self.model_weights.items():
                print(f"  - {model_name}: {weight:.3f}")
        
        def _calculate_model_weights(self, X, y):
            """Вычисляет веса моделей на основе их производительности на валидационном наборе
            
            Args:
                X: Валидационные данные
                y: Валидационные метки
                
            Returns:
                weights: Словарь с весами моделей
            """
            if X is None or y is None:
                return self.model_weights
            
            # Вычисляем точность и F1 для каждой модели
            accuracies = {}
            f1_scores = {}
            
            # Оцениваем основную нейронную сеть
            main_nn = self.models['main_nn']
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                probs = main_nn.predict(X)
                preds = (probs > 0.5).astype(int).flatten()
            else:  # Многоклассовая классификация
                probs = main_nn.predict(X)
                preds = np.argmax(probs, axis=1)
            
            from sklearn.metrics import accuracy_score, f1_score
            nn_accuracy = accuracy_score(y, preds)
            nn_f1 = f1_score(y, preds, average='weighted' if len(np.unique(y)) > 2 else 'binary')
            
            # Используем более агрессивный весовой коэффициент для нейронной сети
            # если она показывает действительно хорошие результаты
            if nn_accuracy > 0.85:
                nn_weight = 0.7
            elif nn_accuracy > 0.75:
                nn_weight = 0.6
            else:
                nn_weight = max(0.5, nn_accuracy)
            
            # Оцениваем вспомогательные модели
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    preds = model.predict(X)
                    acc = accuracy_score(y, preds)
                    
                    # Используем f1-score для более точной оценки на несбалансированных данных
                    if len(np.unique(y)) == 2:  # Бинарная классификация
                        f1 = f1_score(y, preds, average='binary')
                    else:  # Многоклассовая классификация
                        f1 = f1_score(y, preds, average='weighted')
                    
                    # Сохраняем метрики
                    accuracies[name] = acc
                    f1_scores[name] = f1
                except:
                    accuracies[name] = 0.5
                    f1_scores[name] = 0.4
            
            # Комбинируем метрики для лучшей оценки производительности
            performance_scores = {}
            for name in accuracies.keys():
                # Взвешенная сумма точности и F1-меры с акцентом на F1
                performance_scores[name] = 0.4 * accuracies[name] + 0.6 * f1_scores[name]
            
            # Удаляем слабые модели (существенно ниже среднего)
            mean_performance = np.mean(list(performance_scores.values()))
            performance_scores = {name: score for name, score in performance_scores.items() 
                                if score > mean_performance * 0.85}
            
            # Нормализуем веса так, чтобы их сумма была равна 1 - nn_weight
            total = sum(performance_scores.values())
            if total > 0:
                weights = {name: (score / total) * (1 - nn_weight) for name, score in performance_scores.items()}
            else:
                # Если нет производительных моделей, используем нейросеть с большим весом
                weights = {name: (1 - nn_weight) / len(performance_scores) for name in performance_scores.keys()}
            
            # Обеспечиваем минимальный вес для разнообразия
            min_weight = 0.01
            for name in weights:
                if weights[name] < min_weight:
                    weights[name] = min_weight
            
            # Нормализуем веса снова
            total = sum(weights.values())
            weights = {name: weight / total * (1 - nn_weight) for name, weight in weights.items()}
            
            return weights

        def _get_adaptive_threshold(self, X, noise_type=None, noise_level=None):
            """Определяет адаптивный порог уверенности для текущих данных
            
            Args:
                X: Входные данные
                noise_type: Тип шума
                noise_level: Уровень шума
                
            Returns:
                threshold: Адаптивный порог уверенности
            """
            # Базовый порог
            threshold = self.confidence_threshold
            
            # Адаптация на основе типа и уровня шума
            if noise_type and noise_level:
                # Для сильного шума снижаем порог уверенности
                if noise_level > 0.4:
                    threshold -= 0.15
                elif noise_level > 0.2:
                    threshold -= 0.1
                
                # Для специфических типов шума
                if noise_type in ['impulse', 'salt_pepper']:
                    threshold -= 0.05
                elif noise_type == 'missing':
                    threshold -= 0.1
            
            # Убедимся, что порог не стал слишком низким
            threshold = max(0.4, threshold)
            
            return threshold
        
        def _get_model_weights(self, noise_type=None):
            """Возвращает оптимальные веса моделей с учетом типа шума
            
            Args:
                noise_type: Тип шума
                
            Returns:
                weights: Веса моделей
            """
            if noise_type is not None and noise_type in self.noise_type_weights:
                return self.noise_type_weights[noise_type]
            else:
                return self.model_weights
            
        def predict(self, X, noise_type=None, noise_level=None):
            """Делает предсказания с использованием улучшенного адаптивного ансамбля
            
            Args:
                X: Данные для предсказания
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                predictions: Предсказанные метки классов
            """
            # Получаем предсказания основной нейронной сети
            main_nn = self.models['main_nn']
            
            # Проверяем формат выхода (бинарная или многоклассовая классификация)
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.maximum(nn_probs, 1 - nn_probs)  # Уверенность
                nn_preds = (nn_probs > 0.5).astype(int).flatten()
            else:  # Многоклассовая классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
                nn_preds = np.argmax(nn_probs, axis=1)
            
            # Определяем адаптивный порог уверенности
            adaptive_threshold = self._get_adaptive_threshold(X, noise_type, noise_level)
            
            # Находим примеры с низкой уверенностью
            low_conf_mask = nn_conf < adaptive_threshold
            
            # Если все предсказания уверенные, возвращаем их
            if not np.any(low_conf_mask):
                return nn_preds
            
            # Для неуверенных примеров запускаем вспомогательные модели
            X_low_conf = X[low_conf_mask]
            
            # Получаем оптимальные веса моделей для текущего типа шума
            model_weights = self._get_model_weights(noise_type)
            
            # Словарь для хранения работающих моделей и их предсказаний
            working_models = {}
            support_probs = {}
            
            # Получаем предсказания вспомогательных моделей
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    # Предсказания и вероятности
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_low_conf)
                        support_probs[name] = probs
                        working_models[name] = True
                    else:
                        # Пытаемся получить вероятности через решающую функцию
                        try:
                            decision_values = model.decision_function(X_low_conf)
                            # Преобразуем решающую функцию в вероятности с помощью softmax
                            if decision_values.ndim == 1:  # Бинарная классификация
                                probs = 1 / (1 + np.exp(-decision_values))
                                support_probs[name] = np.column_stack([1 - probs, probs])
                            else:  # Многоклассовая классификация
                                exp_decision = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                                probs = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
                                support_probs[name] = probs
                            working_models[name] = True
                        except:
                            # Если ничего не работает, используем one-hot закодированные предсказания
                            preds = model.predict(X_low_conf)
                            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                                probs = np.zeros((len(preds), 2))
                                probs[np.arange(len(preds)), preds] = 1
                            else:  # Многоклассовая классификация
                                probs = np.zeros((len(preds), main_nn.output_shape[-1]))
                                probs[np.arange(len(preds)), preds] = 1
                            support_probs[name] = probs
                            working_models[name] = True
                except Exception as e:
                    print(f"Предупреждение: Ошибка при получении предсказаний от модели {name}: {e}")
                    working_models[name] = False
            
            # Комбинируем предсказания вспомогательных моделей
            final_preds = nn_preds.copy()
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                # Комбинируем вероятности с учетом весов
                weighted_probs = np.zeros((X_low_conf.shape[0], 2))
                model_count = 0
                
                for name, probs in support_probs.items():
                    if working_models.get(name, False) and probs.shape[1] == 2:
                        weight = model_weights.get(name, 0.1)
                        weighted_probs += weight * probs
                        model_count += 1
                
                # Если ни одна модель не сработала, оставляем предсказания нейросети
                if model_count == 0:
                    return nn_preds
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                row_sums = np.where(row_sums == 0, 1, row_sums)  # Избегаем деления на ноль
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Получаем финальные предсказания
                ensemble_preds = np.argmax(normalized_probs, axis=1)
                final_preds[low_conf_mask] = ensemble_preds
                
            else:  # Многоклассовая классификация
                num_classes = main_nn.output_shape[-1]
                weighted_probs = np.zeros((X_low_conf.shape[0], num_classes))
                model_count = 0
                
                for name, probs in support_probs.items():
                    if working_models.get(name, False) and probs.shape[1] == num_classes:
                        weight = model_weights.get(name, 0.1)
                        weighted_probs += weight * probs
                        model_count += 1
                
                # Если ни одна модель не сработала, оставляем предсказания нейросети
                if model_count == 0:
                    return nn_preds
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                row_sums = np.where(row_sums == 0, 1, row_sums)  # Избегаем деления на ноль
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Получаем финальные предсказания
                ensemble_preds = np.argmax(normalized_probs, axis=1)
                final_preds[low_conf_mask] = ensemble_preds
            
            return final_preds
        
        def predict_proba(self, X, noise_type=None, noise_level=None):
            """Предсказывает вероятности классов с учетом всех моделей в ансамбле
            
            Args:
                X: Данные для предсказания
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                probabilities: Предсказанные вероятности классов
            """
            # Получаем вероятности от основной нейронной сети
            main_nn = self.models['main_nn']
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs_raw = main_nn.predict(X)
                nn_probs = np.column_stack([1 - nn_probs_raw, nn_probs_raw])
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
            else:  # Многоклассовая классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
            
            # Определяем адаптивный порог уверенности
            adaptive_threshold = self._get_adaptive_threshold(X, noise_type, noise_level)
            
            # Находим примеры с низкой уверенностью
            low_conf_mask = nn_conf < adaptive_threshold
            
            # Если все предсказания уверенные, возвращаем вероятности от основной модели
            if not np.any(low_conf_mask):
                return nn_probs
            
            # Для неуверенных примеров запускаем вспомогательные модели
            X_low_conf = X[low_conf_mask]
            
            # Получаем оптимальные веса моделей для текущего типа шума
            model_weights = self._get_model_weights(noise_type)
            
            # Словарь для хранения работающих моделей
            working_models = {}
            support_probs = {}
            
            # Получаем вероятности от вспомогательных моделей
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_low_conf)
                        support_probs[name] = probs
                        working_models[name] = True
                    else:
                        try:
                            decision_values = model.decision_function(X_low_conf)
                            if decision_values.ndim == 1:  # Бинарная классификация
                                probs = 1 / (1 + np.exp(-decision_values))
                                support_probs[name] = np.column_stack([1 - probs, probs])
                            else:  # Многоклассовая классификация
                                exp_decision = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                                probs = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
                                support_probs[name] = probs
                            working_models[name] = True
                        except:
                            preds = model.predict(X_low_conf)
                            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                                probs = np.zeros((len(preds), 2))
                                probs[np.arange(len(preds)), preds] = 1
                            else:  # Многоклассовая классификация
                                probs = np.zeros((len(preds), main_nn.output_shape[-1]))
                                probs[np.arange(len(preds)), preds] = 1
                            support_probs[name] = probs
                            working_models[name] = True
                except:
                    working_models[name] = False
            
            # Комбинируем вероятности с учетом весов
            final_probs = nn_probs.copy()
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                weighted_probs = np.zeros((X_low_conf.shape[0], 2))
                model_count = 0
                
                for name, probs in support_probs.items():
                    if working_models.get(name, False) and probs.shape[1] == 2:
                        weight = model_weights.get(name, 0.1)
                        weighted_probs += weight * probs
                        model_count += 1
                
                # Если ни одна модель не сработала, оставляем вероятности нейросети
                if model_count == 0:
                    return nn_probs
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                row_sums = np.where(row_sums == 0, 1, row_sums)  # Избегаем деления на ноль
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Обновляем финальные вероятности
                final_probs[low_conf_mask] = normalized_probs
                
            else:  # Многоклассовая классификация
                num_classes = main_nn.output_shape[-1]
                weighted_probs = np.zeros((X_low_conf.shape[0], num_classes))
                model_count = 0
                
                for name, probs in support_probs.items():
                    if working_models.get(name, False) and probs.shape[1] == num_classes:
                        weight = model_weights.get(name, 0.1)
                        weighted_probs += weight * probs
                        model_count += 1
                
                # Если ни одна модель не сработала, оставляем вероятности нейросети
                if model_count == 0:
                    return nn_probs
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                row_sums = np.where(row_sums == 0, 1, row_sums)  # Избегаем деления на ноль
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Обновляем финальные вероятности
                final_probs[low_conf_mask] = normalized_probs
            
            return final_probs
            
        def evaluate(self, X, y, noise_type=None, noise_level=None):
            """Оценивает производительность ансамбля
            
            Args:
                X: Тестовые данные
                y: Истинные метки
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                metrics: Словарь с метриками производительности
            """
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
            
            # Делаем предсказания
            y_pred = self.predict(X, noise_type, noise_level)
            y_proba = self.predict_proba(X, noise_type, noise_level)
            
            # Вычисляем метрики
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            
            # Дополнительные метрики
            if len(np.unique(y)) == 2:  # Бинарная классификация
                f1 = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')
            else:  # Многоклассовая классификация
                f1 = f1_score(y, y_pred, average='weighted')
                precision = precision_score(y, y_pred, average='weighted')
                recall = recall_score(y, y_pred, average='weighted')
            
            # Оцениваем производительность отдельных моделей
            models_metrics = {}
            for name, model in self.models.items():
                try:
                    if name == 'main_nn':
                        if model.output_shape[-1] == 1:  # Бинарная классификация
                            probs = model.predict(X)
                            preds = (probs > 0.5).astype(int).flatten()
                        else:  # Многоклассовая классификация
                            probs = model.predict(X)
                            preds = np.argmax(probs, axis=1)
                    else:
                        preds = model.predict(X)
                    
                    model_acc = accuracy_score(y, preds)
                    if len(np.unique(y)) == 2:  # Бинарная классификация
                        model_f1 = f1_score(y, preds, average='binary')
                    else:  # Многоклассовая классификация
                        model_f1 = f1_score(y, preds, average='weighted')
                        
                    models_metrics[name] = {
                        'accuracy': model_acc,
                        'f1_score': model_f1
                    }
                except:
                    models_metrics[name] = {'accuracy': 0.0, 'f1_score': 0.0}
            
            # Возвращаем метрики
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'report': report,
                'models_metrics': models_metrics
            }

class ExperimentRunner:
    """Класс для проведения экспериментов с моделями классификации на зашумленных данных"""
    
    def __init__(self, dataset_name=None, dataset_path=None):
        """Инициализирует средство проведения экспериментов
        
        Args:
            dataset_name: Название набора данных из sklearn (если используется встроенный набор)
            dataset_path: Путь к файлу с набором данных (если используется внешний набор)
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.noise_injector = NoiseInjector()
        self.noise_preprocessor = NoisePreprocessor()
        self.model_builder = ModelBuilder()
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.scaler = RobustScaler()  # Более устойчив к выбросам
        self.experiment_results = {}
        self.current_ensemble = None
        
    def load_dataset(self, dataset_name=None, dataset_path=None):
        """Загружает набор данных
        
        Args:
            dataset_name: Название набора данных из sklearn (если используется встроенный набор)
            dataset_path: Путь к файлу с набором данных (если используется внешний набор)
            
        Returns:
            X: Признаки
            y: Метки классов
        """
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if dataset_path is not None:
            self.dataset_path = dataset_path
            
        # Загрузка встроенных наборов данных
        if self.dataset_name == 'iris':
            data = load_iris()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Iris: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'wine':
            data = load_wine()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Wine: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Breast Cancer: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'digits':
            data = fetch_openml('mnist_784', version=1, parser='auto')
            # Для ускорения используем только часть набора данных MNIST
            n_samples = 5000
            self.X = data.data[:n_samples].astype(float).values
            self.y = data.target[:n_samples].astype(int).values
            self.feature_names = [f"pixel_{i}" for i in range(self.X.shape[1])]
            self.target_names = [str(i) for i in range(10)]
            print(f"Загружен набор данных MNIST (подвыборка): {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        # Загрузка внешнего набора данных
        elif self.dataset_path is not None:
            if self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path)
                
                # Предполагаем, что последний столбец - это метки классов
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                self.X = X
                self.y = y
                self.feature_names = df.columns[:-1].tolist()
                self.target_names = [str(label) for label in np.unique(y)]
                
                print(f"Загружен пользовательский набор данных: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            else:
                raise ValueError("Поддерживаются только файлы CSV")
                
        else:
            raise ValueError("Необходимо указать название набора данных или путь к файлу")
        
        return self.X, self.y
    
    # Обновите run_experiment в классе ExperimentRunner
    def run_experiment(self, noise_type, noise_range, noise_step, n_experiments=3, use_preprocessing=True):
        """Проводит эксперимент с заданным типом и уровнем шума
        
        Args:
            noise_type: Тип шума ('gaussian', 'uniform', 'impulse', 'missing', 'salt_pepper', 'multiplicative')
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            use_preprocessing: Применять ли предобработку данных в зависимости от типа шума
            
        Returns:
            results: Словарь с результатами экспериментов
        """
        if self.X is None or self.y is None:
            raise ValueError("Набор данных не загружен")
        
        # Словарь для хранения результатов
        results = {
            'noise_levels': [],
            'ensemble_accuracy': [],
            'ensemble_f1': [],
            'nn_accuracy': [],
            'rf_accuracy': [],
            'gb_accuracy': [],
            'svm_accuracy': [],
            'knn_accuracy': [],
            'xgb_accuracy': [],
            'lgb_accuracy': [],
            'preprocessing_impact': []  # Новый ключ для хранения влияния предобработки
        }
        
        min_noise, max_noise = noise_range
        noise_levels = np.arange(min_noise, max_noise + noise_step, noise_step)
        
        # Предварительная обработка данных
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Проверяем на дисбаланс классов
        class_counts = np.bincount(self.y)
        min_class_count = np.min(class_counts)
        max_class_count = np.max(class_counts)
        class_imbalance_ratio = max_class_count / min_class_count
        
        # Если имеется сильный дисбаланс классов, применяем SMOTETomek
        use_smote = class_imbalance_ratio > 2.5
        if use_smote:
            print(f"\nОбнаружен дисбаланс классов (соотношение: {class_imbalance_ratio:.2f}). Применение SMOTETomek...")
        
        # Количество классов
        num_classes = len(np.unique(self.y))
        input_shape = (self.X.shape[1],)
        
        # Разбиение данных со стратификацией
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        
        # Применяем SMOTETomek если необходимо
        if use_smote:
            smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
            X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
            print(f"После SMOTETomek: {X_train.shape[0]} образцов, распределение классов: {np.bincount(y_train)}")
        
        print(f"\nПроводим эксперимент с шумом типа {noise_type}...")
        print(f"Диапазон шума: [{min_noise}, {max_noise}], шаг: {noise_step}")
        print(f"Количество экспериментов для усреднения: {n_experiments}")
        print(f"Применение предобработки шума: {use_preprocessing}")
        
        # Выполняем отбор признаков (если признаков много)
        if X_train.shape[1] > 10:
            print("\nВыполняем отбор признаков...")
            X_train_selected = self.model_builder.perform_feature_selection(X_train, y_train)
            X_val_selected = self.model_builder.apply_feature_transformation(X_val)
            X_test_selected = self.model_builder.apply_feature_transformation(X_test)
            
            # Обновляем размерность входных данных
            input_shape = (X_train_selected.shape[1],)
        else:
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test
        
        # Оптимизация гиперпараметров основной нейронной сети
        print("\nОптимизация гиперпараметров основной нейронной сети...")
        nn_params = self.model_builder.optimize_neural_network(
            X_train_selected, y_train, X_val_selected, y_val, input_shape, num_classes, 
            n_trials=30, noise_type=noise_type  # n_trials - попытки оптимизации гиперпараметров нейронной сети
        )
        
        # Оптимизация гиперпараметров вспомогательных моделей
        print("\nОптимизация гиперпараметров вспомогательных моделей...")
        support_params = self.model_builder.optimize_support_models(X_train_selected, y_train)
        
        # Создаем ансамблевую модель
        print("\nСоздание ансамблевой модели...")
        models = self.model_builder.build_ensemble_model(
            input_shape, num_classes, nn_params, support_params
        )
        
        # Обучаем основную нейронную сеть
        print("\nОбучение основной нейронной сети...")
        if num_classes > 2:
            y_train_cat = to_categorical(y_train)
            y_val_cat = to_categorical(y_val)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
            
        # Callback для сохранения лучшей модели
        checkpoint = ModelCheckpoint(
            'best_nn_model',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True  # Увеличиваем patience
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6  # Более агрессивное снижение LR
        )
        
        # Обучаем с увеличенным количеством эпох
        models['main_nn'].fit(
            X_train_selected, y_train_cat,
            epochs=150,  # Увеличиваем количество эпох
            batch_size=nn_params['batch_size'],
            validation_data=(X_val_selected, y_val_cat),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Загружаем лучшую модель (если сохранялась)
        custom_objects = {'FocalLoss': FocalLoss, 'CategoricalFocalLoss': CategoricalFocalLoss}
        if os.path.exists('best_nn_model'):
            models['main_nn'] = keras.models.load_model('best_nn_model', custom_objects=custom_objects)
            print("Загружена лучшая модель нейронной сети")
        
        # Обучаем вспомогательные модели
        print("\nОбучение вспомогательных моделей...")
        for name, model in models.items():
            if name != 'main_nn':
                model.fit(X_train_selected, y_train)
        
        # Создаем улучшенный адаптивный ансамбль с калибровкой весов
        print("\nСоздание улучшенного адаптивного ансамбля...")
        ensemble = self.model_builder.ImprovedAdaptiveEnsemble(models, X_val_selected, y_val)
        self.current_ensemble = ensemble
        
        # Проводим эксперименты для каждого уровня шума
        for noise_level in noise_levels:
            print(f"\nТестирование с уровнем шума {noise_level:.3f}...")
            
            # Массивы для хранения результатов экспериментов
            ensemble_accs = []
            ensemble_f1s = []
            nn_accs = []
            rf_accs = []
            gb_accs = []
            svm_accs = []
            knn_accs = []
            xgb_accs = []
            lgb_accs = []
            preprocessing_impacts = []  # Для измерения эффекта предобработки
            
            for exp in range(n_experiments):
                print(f"Эксперимент {exp + 1}/{n_experiments}...")
                
                # Добавляем шум к тестовым данным
                if noise_type == 'gaussian':
                    X_test_noisy = self.noise_injector.add_gaussian_noise(X_test_selected, noise_level)
                elif noise_type == 'uniform':
                    X_test_noisy = self.noise_injector.add_uniform_noise(X_test_selected, noise_level)
                elif noise_type == 'impulse':
                    X_test_noisy = self.noise_injector.add_impulse_noise(X_test_selected, noise_level)
                elif noise_type == 'missing':
                    X_test_noisy = self.noise_injector.add_missing_values(X_test_selected, noise_level)
                    # Базовая импутация для работоспособности, дополнительная будет в предобработке
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X_test_noisy = imputer.fit_transform(X_test_noisy)
                elif noise_type == 'salt_pepper':
                    X_test_noisy = self.noise_injector.add_salt_pepper_noise(X_test_selected, noise_level)
                elif noise_type == 'multiplicative':
                    X_test_noisy = self.noise_injector.add_multiplicative_noise(X_test_selected, noise_level)
                else:
                    raise ValueError(f"Неизвестный тип шума: {noise_type}")
                
                # Делаем копию для оценки без предобработки
                X_test_raw = X_test_noisy.copy()
                
                # Применяем предобработку в зависимости от типа шума
                if use_preprocessing:
                    # Используем улучшенный предпроцессор
                    X_test_preprocessed = self.noise_preprocessor.preprocess_data(X_test_noisy, noise_type)
                    
                    # Оцениваем эффект предобработки
                    ensemble_metrics_raw = ensemble.evaluate(X_test_raw, y_test, noise_type, noise_level)
                    ensemble_metrics_preprocessed = ensemble.evaluate(X_test_preprocessed, y_test, noise_type, noise_level)
                    
                    # Сравниваем точность до и после предобработки
                    acc_raw = ensemble_metrics_raw['accuracy']
                    acc_preprocessed = ensemble_metrics_preprocessed['accuracy']
                    preprocessing_impact = acc_preprocessed - acc_raw
                    preprocessing_impacts.append(preprocessing_impact)
                    
                    # Используем предобработанные данные
                    X_test_final = X_test_preprocessed
                    print(f"  Влияние предобработки: {preprocessing_impact*100:.2f}% ({acc_raw:.4f} -> {acc_preprocessed:.4f})")
                else:
                    X_test_final = X_test_raw
                    preprocessing_impacts.append(0.0)
                
                # Оцениваем ансамбль
                metrics = ensemble.evaluate(X_test_final, y_test, noise_type, noise_level)
                
                # Сохраняем результаты
                ensemble_accs.append(metrics['accuracy'])
                ensemble_f1s.append(metrics['f1_score'])
                nn_accs.append(metrics['models_metrics']['main_nn']['accuracy'])
                rf_accs.append(metrics['models_metrics']['random_forest']['accuracy'])
                gb_accs.append(metrics['models_metrics']['gradient_boosting']['accuracy'])
                svm_accs.append(metrics['models_metrics']['svm']['accuracy'])
                knn_accs.append(metrics['models_metrics']['knn']['accuracy'])
                xgb_accs.append(metrics['models_metrics']['xgboost']['accuracy'])
                lgb_accs.append(metrics['models_metrics']['lightgbm']['accuracy'])
            
            # Вычисляем средние значения и стандартные отклонения
            results['noise_levels'].append(noise_level)
            results['ensemble_accuracy'].append((np.mean(ensemble_accs), np.std(ensemble_accs)))
            results['ensemble_f1'].append((np.mean(ensemble_f1s), np.std(ensemble_f1s)))
            results['nn_accuracy'].append((np.mean(nn_accs), np.std(nn_accs)))
            results['rf_accuracy'].append((np.mean(rf_accs), np.std(rf_accs)))
            results['gb_accuracy'].append((np.mean(gb_accs), np.std(gb_accs)))
            results['svm_accuracy'].append((np.mean(svm_accs), np.std(svm_accs)))
            results['knn_accuracy'].append((np.mean(knn_accs), np.std(knn_accs)))
            results['xgb_accuracy'].append((np.mean(xgb_accs), np.std(xgb_accs)))
            results['lgb_accuracy'].append((np.mean(lgb_accs), np.std(lgb_accs)))
            results['preprocessing_impact'].append((np.mean(preprocessing_impacts), np.std(preprocessing_impacts)))
            
            print(f"Средняя точность ансамбля: {np.mean(ensemble_accs):.4f} ± {np.std(ensemble_accs):.4f}")
            print(f"Средняя F1-мера ансамбля: {np.mean(ensemble_f1s):.4f} ± {np.std(ensemble_f1s):.4f}")
            print(f"Средняя точность нейронной сети: {np.mean(nn_accs):.4f} ± {np.std(nn_accs):.4f}")
            print(f"Средняя точность Random Forest: {np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}")
            print(f"Средняя точность Gradient Boosting: {np.mean(gb_accs):.4f} ± {np.std(gb_accs):.4f}")
            print(f"Средняя точность SVM: {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
            print(f"Средняя точность KNN: {np.mean(knn_accs):.4f} ± {np.std(knn_accs):.4f}")
            print(f"Средняя точность XGBoost: {np.mean(xgb_accs):.4f} ± {np.std(xgb_accs):.4f}")
            print(f"Средняя точность LightGBM: {np.mean(lgb_accs):.4f} ± {np.std(lgb_accs):.4f}")
            if use_preprocessing:
                print(f"Среднее влияние предобработки: {np.mean(preprocessing_impacts)*100:.2f}% ± {np.std(preprocessing_impacts)*100:.2f}%")
        
        # Сохраняем результаты эксперимента
        self.experiment_results[noise_type] = results
        
        return results
    
    def run_all_experiments(self, noise_range, noise_step, n_experiments=3, use_preprocessing=True):
        """Проводит все эксперименты с различными типами шума
        
        Args:
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            use_preprocessing: Применять ли предобработку данных в зависимости от типа шума
            
        Returns:
            all_results: Словарь с результатами всех экспериментов
        """
        noise_types = ['gaussian', 'uniform', 'impulse', 'missing', 'salt_pepper', 'multiplicative']
        all_results = {}
        
        for noise_type in noise_types:
            print(f"\n{'=' * 50}")
            print(f"Запуск экспериментов с шумом типа {noise_type}")
            print(f"{'=' * 50}")
            
            results = self.run_experiment(noise_type, noise_range, noise_step, n_experiments, use_preprocessing)
            all_results[noise_type] = results
        
        self.experiment_results = all_results
        return all_results
    
    def visualize_results(self, noise_type=None, show_preprocessing=True, metric='accuracy', figsize=(12, 8)):
        """Визуализирует результаты экспериментов
        
        Args:
            noise_type: Тип шума для визуализации (если None, визуализируются все)
            show_preprocessing: Показывать ли влияние предобработки
            metric: Метрика для визуализации ('accuracy' или 'f1')
            figsize: Размер фигуры
            
        Returns:
            fig: Объект фигуры matplotlib
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для визуализации")
        
        if noise_type is not None:
            if noise_type not in self.experiment_results:
                raise ValueError(f"Нет результатов для шума типа {noise_type}")
            
            # Визуализация результатов для одного типа шума
            results = self.experiment_results[noise_type]
            
            fig, ax = plt.subplots(figsize=figsize)
            
            noise_levels = results['noise_levels']
            
            # Настройка стилей
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('seaborn')
            
            # Точность ансамбля
            if metric == 'accuracy':
                ensemble_mean = [acc[0] for acc in results['ensemble_accuracy']]
                ensemble_std = [acc[1] for acc in results['ensemble_accuracy']]
                metric_label = 'Точность'
            else:  # f1
                ensemble_mean = [f1[0] for f1 in results['ensemble_f1']]
                ensemble_std = [f1[1] for f1 in results['ensemble_f1']]
                metric_label = 'F1-мера'
            
            ax.plot(noise_levels, ensemble_mean, 'o-', linewidth=2, color='#1f77b4', label='Ансамблевая модель')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(ensemble_mean, ensemble_std)],
                            [m + s for m, s in zip(ensemble_mean, ensemble_std)],
                            alpha=0.2, color='#1f77b4')
            
            # Точность основной нейронной сети
            nn_mean = [acc[0] for acc in results['nn_accuracy']]
            nn_std = [acc[1] for acc in results['nn_accuracy']]
            ax.plot(noise_levels, nn_mean, 's-', linewidth=2, color='#d62728', label='Нейронная сеть')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(nn_mean, nn_std)],
                            [m + s for m, s in zip(nn_mean, nn_std)],
                            alpha=0.2, color='#d62728')
            
            # Точность остальных моделей
            rf_mean = [acc[0] for acc in results['rf_accuracy']]
            gb_mean = [acc[0] for acc in results['gb_accuracy']]
            svm_mean = [acc[0] for acc in results['svm_accuracy']]
            knn_mean = [acc[0] for acc in results['knn_accuracy']]
            xgb_mean = [acc[0] for acc in results['xgb_accuracy']]
            lgb_mean = [acc[0] for acc in results['lgb_accuracy']]
            
            # Используем более приятные цвета
            ax.plot(noise_levels, rf_mean, '^-', linewidth=2, color='#2ca02c', label='Random Forest')
            ax.plot(noise_levels, gb_mean, 'v-', linewidth=2, color='#ff7f0e', label='Gradient Boosting')
            ax.plot(noise_levels, svm_mean, 'D-', linewidth=2, color='#9467bd', label='SVM')
            ax.plot(noise_levels, knn_mean, 'p-', linewidth=2, color='#8c564b', label='K-NN')
            ax.plot(noise_levels, xgb_mean, '*-', linewidth=2, color='#e377c2', label='XGBoost')
            ax.plot(noise_levels, lgb_mean, 'X-', linewidth=2, color='#7f7f7f', label='LightGBM')
            
            # Если выбран показ влияния предобработки и оно есть в результатах
            if show_preprocessing and 'preprocessing_impact' in results:
                # Создаем вторую ось Y
                ax2 = ax.twinx()
                prep_mean = [impact[0] * 100 for impact in results['preprocessing_impact']]  # В процентах
                prep_std = [impact[1] * 100 for impact in results['preprocessing_impact']]
                
                ax2.plot(noise_levels, prep_mean, '--', linewidth=2, color='#17becf', label='Влияние предобработки')
                ax2.fill_between(noise_levels,
                                [m - s for m, s in zip(prep_mean, prep_std)],
                                [m + s for m, s in zip(prep_mean, prep_std)],
                                alpha=0.2, color='#17becf')
                
                # Настройки вторичной оси Y
                ax2.set_ylabel('Изменение точности после предобработки, %')
                ax2.spines['right'].set_color('#17becf')
                ax2.yaxis.label.set_color('#17becf')
                ax2.tick_params(axis='y', colors='#17becf')
                
                # Добавляем легенду для второй оси
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            # Настройка осей и заголовка
            ax.set_xlabel('Уровень шума')
            ax.set_ylabel(metric_label)
            ax.set_title(f'Зависимость {metric_label.lower()} от уровня шума типа {noise_type}')
            
            if not show_preprocessing or 'preprocessing_impact' not in results:
                ax.legend(loc='best')
            
            ax.grid(True, alpha=0.3)
            
            # Настройка внешнего вида графика
            plt.tight_layout()
            
            return fig
            
        else:
            # Визуализация сравнения результатов для всех типов шума
            # Определим количество типов шума для визуализации
            noise_types_to_plot = [nt for nt in self.experiment_results.keys()]
            n_noise_types = len(noise_types_to_plot)
            
            # Определим размер сетки для графиков (стараемся сделать её более квадратной)
            n_cols = min(3, n_noise_types)  # Максимум 3 графика в ширину
            n_rows = (n_noise_types + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            
            # Преобразуем массив осей в плоский список для удобства
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = np.array(axes).flatten()
            
            # Настройка стилей для всех графиков
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('seaborn')
            
            # Цветовая схема
            colors = {
                'ensemble': '#1f77b4',
                'nn': '#d62728',
                'rf': '#2ca02c',
                'gb': '#ff7f0e',
                'svm': '#9467bd',
                'knn': '#8c564b',
                'xgb': '#e377c2',
                'lgb': '#7f7f7f'
            }
            
            for i, noise_type in enumerate(noise_types_to_plot):
                if i >= len(axes):
                    break
                    
                results = self.experiment_results[noise_type]
                ax = axes[i]
                
                noise_levels = results['noise_levels']
                
                # Точность ансамбля
                if metric == 'accuracy':
                    ensemble_mean = [acc[0] for acc in results['ensemble_accuracy']]
                    ensemble_std = [acc[1] for acc in results['ensemble_accuracy']]
                    metric_label = 'Точность'
                else:  # f1
                    ensemble_mean = [f1[0] for f1 in results['ensemble_f1']]
                    ensemble_std = [f1[1] for f1 in results['ensemble_f1']]
                    metric_label = 'F1-мера'
                
                ax.plot(noise_levels, ensemble_mean, 'o-', linewidth=2, color=colors['ensemble'], label='Ансамбль')
                ax.fill_between(noise_levels, 
                                [m - s for m, s in zip(ensemble_mean, ensemble_std)],
                                [m + s for m, s in zip(ensemble_mean, ensemble_std)],
                                alpha=0.2, color=colors['ensemble'])
                
                # Точность основной нейронной сети
                nn_mean = [acc[0] for acc in results['nn_accuracy']]
                ax.plot(noise_levels, nn_mean, 's-', linewidth=2, color=colors['nn'], label='Нейросеть')
                
                # Точность остальных моделей (упрощаем для лучшей читаемости)
                rf_mean = [acc[0] for acc in results['rf_accuracy']]
                gb_mean = [acc[0] for acc in results['gb_accuracy']]
                
                # Добавим только основные модели для ясности
                ax.plot(noise_levels, rf_mean, '^-', linewidth=2, color=colors['rf'], label='Random Forest')
                ax.plot(noise_levels, gb_mean, 'v-', linewidth=2, color=colors['gb'], label='Gradient Boost')
                
                # Если включены дополнительные модели, добавим основные из них
                if 'xgb_accuracy' in results:
                    xgb_mean = [acc[0] for acc in results['xgb_accuracy']]
                    ax.plot(noise_levels, xgb_mean, '*-', linewidth=2, color=colors['xgb'], label='XGBoost')
                
                ax.set_xlabel('Уровень шума')
                ax.set_ylabel(metric_label)
                ax.set_title(f'Шум типа {noise_type}')
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, alpha=0.3)
            
            # Скрываем пустые подграфики
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def visualize_single_noise_level(self, noise_type, noise_level_idx=None, figsize=(10, 6)):
        """Визуализирует сравнение моделей для конкретного уровня шума
        
        Args:
            noise_type: Тип шума для визуализации
            noise_level_idx: Индекс уровня шума (если None, берется последний уровень)
            figsize: Размер фигуры
            
        Returns:
            fig: Объект фигуры matplotlib
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для визуализации")
        
        if noise_type not in self.experiment_results:
            raise ValueError(f"Нет результатов для шума типа {noise_type}")
        
        results = self.experiment_results[noise_type]
        noise_levels = results['noise_levels']
        
        if noise_level_idx is None:
            noise_level_idx = len(noise_levels) - 1  # Последний уровень шума
        
        if noise_level_idx < 0 or noise_level_idx >= len(noise_levels):
            raise ValueError(f"Индекс уровня шума должен быть в диапазоне [0, {len(noise_levels)-1}]")
        
        noise_level = noise_levels[noise_level_idx]
        
        # Собираем данные моделей
        models_data = {
            'Ансамбль': results['ensemble_accuracy'][noise_level_idx],
            'Нейронная сеть': results['nn_accuracy'][noise_level_idx],
            'Random Forest': results['rf_accuracy'][noise_level_idx],
            'Gradient Boosting': results['gb_accuracy'][noise_level_idx],
            'SVM': results['svm_accuracy'][noise_level_idx],
            'KNN': results['knn_accuracy'][noise_level_idx]
        }
        
        if 'xgb_accuracy' in results:
            models_data['XGBoost'] = results['xgb_accuracy'][noise_level_idx]
        
        if 'lgb_accuracy' in results:
            models_data['LightGBM'] = results['lgb_accuracy'][noise_level_idx]
        
        # Сортируем модели по точности
        sorted_models = sorted(models_data.items(), key=lambda x: x[1][0], reverse=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настройка стилей
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        # Цвета для моделей
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
        
        # Строим бар-график с ошибками
        model_names = [model[0] for model in sorted_models]
        accuracies = [model[1][0] for model in sorted_models]
        errors = [model[1][1] for model in sorted_models]
        
        bars = ax.bar(model_names, accuracies, yerr=errors, capsize=5, color=colors, alpha=0.7)
        
        # Добавляем значения над столбцами
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Настройки оси и заголовка
        ax.set_xlabel('Модели')
        ax.set_ylabel('Точность')
        ax.set_title(f'Сравнение моделей при шуме типа {noise_type}, уровень {noise_level:.2f}')
        ax.set_ylim(0, min(1.0, max(accuracies) + 0.1))
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def visualize_preprocessing_impact(self, figsize=(12, 8)):
        """Визуализирует влияние предобработки данных на точность для всех типов шума
        
        Args:
            figsize: Размер фигуры
            
        Returns:
            fig: Объект фигуры matplotlib
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для визуализации")
        
        noise_types = list(self.experiment_results.keys())
        
        # Проверяем наличие информации о предобработке
        has_preprocessing_data = all('preprocessing_impact' in self.experiment_results[nt] for nt in noise_types)
        
        if not has_preprocessing_data:
            raise ValueError("Отсутствуют данные о влиянии предобработки. Запустите эксперименты с параметром use_preprocessing=True")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настройка стилей
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        # Цветовая схема для типов шума
        colors = plt.cm.tab10(np.linspace(0, 1, len(noise_types)))
        
        # Построение графиков для каждого типа шума
        for i, noise_type in enumerate(noise_types):
            results = self.experiment_results[noise_type]
            noise_levels = results['noise_levels']
            
            # Влияние предобработки (в процентах)
            prep_mean = [impact[0] * 100 for impact in results['preprocessing_impact']]
            prep_std = [impact[1] * 100 for impact in results['preprocessing_impact']]
            
            ax.plot(noise_levels, prep_mean, 'o-', linewidth=2, color=colors[i], label=f'Шум типа {noise_type}')
            ax.fill_between(noise_levels,
                            [m - s for m, s in zip(prep_mean, prep_std)],
                            [m + s for m, s in zip(prep_mean, prep_std)],
                            alpha=0.2, color=colors[i])
        
        # Настройка осей и заголовка
        ax.set_xlabel('Уровень шума')
        ax.set_ylabel('Улучшение точности после предобработки, %')
        ax.set_title('Влияние предобработки данных на точность классификации')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)  # Нулевая линия
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """Генерирует отчет о результатах экспериментов в виде таблицы
        
        Returns:
            report_df: DataFrame с результатами
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для отчета")
        
        # Создаем список для хранения данных отчета
        report_data = []
        
        for noise_type, results in self.experiment_results.items():
            noise_levels = results['noise_levels']
            
            for i, level in enumerate(noise_levels):
                # Получаем средние значения и стандартные отклонения
                ensemble_acc = results['ensemble_accuracy'][i]
                
                # Получаем F1-меру ансамбля, если она доступна
                if 'ensemble_f1' in results:
                    ensemble_f1 = results['ensemble_f1'][i]
                    f1_str = f"{ensemble_f1[0]:.4f} ± {ensemble_f1[1]:.4f}"
                else:
                    f1_str = "N/A"
                
                nn_acc = results['nn_accuracy'][i]
                rf_acc = results['rf_accuracy'][i]
                gb_acc = results['gb_accuracy'][i]
                svm_acc = results['svm_accuracy'][i]
                knn_acc = results['knn_accuracy'][i]
                
                # Добавляем данные XGBoost и LightGBM, если доступны
                xgb_str = "N/A"
                lgb_str = "N/A"
                
                if 'xgb_accuracy' in results:
                    xgb_acc = results['xgb_accuracy'][i]
                    xgb_str = f"{xgb_acc[0]:.4f} ± {xgb_acc[1]:.4f}"
                
                if 'lgb_accuracy' in results:
                    lgb_acc = results['lgb_accuracy'][i]
                    lgb_str = f"{lgb_acc[0]:.4f} ± {lgb_acc[1]:.4f}"
                
                # Добавляем информацию о влиянии предобработки, если доступна
                preprocessing_str = "N/A"
                if 'preprocessing_impact' in results:
                    prep_impact = results['preprocessing_impact'][i]
                    preprocessing_str = f"{prep_impact[0]*100:.2f}% ± {prep_impact[1]*100:.2f}%"
                
                # Добавляем данные в отчет
                report_data.append({
                    'Тип шума': noise_type,
                    'Уровень шума': level,
                    'Ансамблевая модель': f"{ensemble_acc[0]:.4f} ± {ensemble_acc[1]:.4f}",
                    'F1-мера ансамбля': f1_str,
                    'Нейронная сеть': f"{nn_acc[0]:.4f} ± {nn_acc[1]:.4f}",
                    'Random Forest': f"{rf_acc[0]:.4f} ± {rf_acc[1]:.4f}",
                    'Gradient Boosting': f"{gb_acc[0]:.4f} ± {gb_acc[1]:.4f}",
                    'SVM': f"{svm_acc[0]:.4f} ± {svm_acc[1]:.4f}",
                    'K-NN': f"{knn_acc[0]:.4f} ± {knn_acc[1]:.4f}",
                    'XGBoost': xgb_str,
                    'LightGBM': lgb_str,
                    'Эффект предобработки': preprocessing_str
                })
        
        # Создаем DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df
    
    def save_models(self, path='./models'):
        """Сохраняет обученные модели
        
        Args:
            path: Путь для сохранения моделей
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Сохраняем объект скалера
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Сохраняем объекты для преобразования признаков, если они есть
        if self.model_builder.feature_selector is not None:
            joblib.dump(self.model_builder.feature_selector, os.path.join(path, 'feature_selector.pkl'))
        
        if self.model_builder.pca is not None:
            joblib.dump(self.model_builder.pca, os.path.join(path, 'pca.pkl'))
        
        # Получаем модели из model_builder
        models = self.model_builder.models
        
        # Сохраняем модели
        if 'main_nn' in models:
            models['main_nn'].save(os.path.join(path, 'main_nn_model'))
        
        for name, model in models.items():
            if name != 'main_nn':
                try:
                    joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
                except Exception as e:
                    print(f"Ошибка при сохранении модели {name}: {e}")
        
        # Сохраняем гиперпараметры
        with open(os.path.join(path, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(self.model_builder.best_params, f)
        
        # Сохраняем текущий ансамбль, если он существует
        if self.current_ensemble is not None:
            try:
                with open(os.path.join(path, 'ensemble_weights.pkl'), 'wb') as f:
                    pickle.dump(self.current_ensemble.model_weights, f)
            except Exception as e:
                print(f"Ошибка при сохранении весов ансамбля: {e}")
        
        print(f"Модели успешно сохранены в директории {path}")
        
    def save_figure(self, fig, filename, formats=None):
        """Сохраняет фигуру в различных форматах
        
        Args:
            fig: Объект фигуры matplotlib
            filename: Имя файла без расширения
            formats: Список форматов для сохранения (по умолчанию ['png', 'pdf', 'svg'])
        """
        if formats is None:
            formats = ['png', 'pdf', 'svg']
        
        # Убедимся, что директория существует
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        for fmt in formats:
            try:
                fig.savefig(f"{filename}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
                print(f"График сохранен в формате {fmt}: {filename}.{fmt}")
            except Exception as e:
                print(f"Ошибка при сохранении в формате {fmt}: {e}")
    
    # Обновите метод load_models в классе ExperimentRunner
    def load_models(self, path='./models'):
        """Загружает обученные модели
        
        Args:
            path: Путь к сохраненным моделям
            
        Returns:
            loaded_models: Словарь с загруженными моделями
        """
        if not os.path.exists(path):
            raise ValueError(f"Директория {path} не существует")
        
        # Загружаем объект скалера
        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Загружаем объекты для преобразования признаков, если они есть
        feature_selector_path = os.path.join(path, 'feature_selector.pkl')
        if os.path.exists(feature_selector_path):
            self.model_builder.feature_selector = joblib.load(feature_selector_path)
        
        pca_path = os.path.join(path, 'pca.pkl')
        if os.path.exists(pca_path):
            self.model_builder.pca = joblib.load(pca_path)
        
        # Загружаем модели
        models = {}
        
        # Загружаем основную нейронную сеть
        nn_path = os.path.join(path, 'main_nn_model')
        if os.path.exists(nn_path):
            try:
                # Определяем словарь с пользовательскими объектами для загрузки
                custom_objects = {
                    'FocalLoss': FocalLoss,
                    'CategoricalFocalLoss': CategoricalFocalLoss
                }
                models['main_nn'] = keras.models.load_model(nn_path, custom_objects=custom_objects)
                print("Загружена основная нейронная сеть")
            except Exception as e:
                print(f"Ошибка при загрузке нейронной сети: {e}")
        
        # Загружаем вспомогательные модели
        model_names = [
            'random_forest', 'gradient_boosting', 'svm', 'knn', 
            'xgboost', 'lightgbm', 'adaboost', 'extra_trees', 'stacking'
        ]
        
        for name in model_names:
            model_path = os.path.join(path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    models[name] = joblib.load(model_path)
                    print(f"Загружена модель {name}")
                except Exception as e:
                    print(f"Ошибка при загрузке модели {name}: {e}")
        
        # Загружаем гиперпараметры
        hyperparams_path = os.path.join(path, 'hyperparameters.pkl')
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'rb') as f:
                self.model_builder.best_params = pickle.load(f)
        
        # Устанавливаем загруженные модели
        self.model_builder.models = models
        
        # Загружаем веса ансамбля, если они есть
        ensemble_weights_path = os.path.join(path, 'ensemble_weights.pkl')
        if os.path.exists(ensemble_weights_path):
            try:
                with open(ensemble_weights_path, 'rb') as f:
                    ensemble_weights = pickle.load(f)
                    
                # Создаем адаптивный ансамбль с загруженными весами
                self.current_ensemble = self.model_builder.ImprovedAdaptiveEnsemble(models)
                self.current_ensemble.model_weights = ensemble_weights
                print("Загружены веса ансамбля")
            except Exception as e:
                print(f"Ошибка при загрузке весов ансамбля: {e}")
                # Создаем новый ансамбль с дефолтными весами
                self.current_ensemble = self.model_builder.ImprovedAdaptiveEnsemble(models)
        else:
            # Создаем новый ансамбль с дефолтными весами
            self.current_ensemble = self.model_builder.ImprovedAdaptiveEnsemble(models)
        
        print("Модели успешно загружены")
        
        return models

class NoisyDataClassificationApp:
    """Класс для создания графического интерфейса программного комплекса"""
    
    def __init__(self, root):
        """Инициализирует приложение
        
        Args:
            root: Корневой виджет Tkinter
        """
        self.root = root
        self.root.title("Программный комплекс для классификации зашумленных данных")
        self.root.geometry("1280x800")
        
        # Устанавливаем иконку, если доступна
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Создаем экземпляр ExperimentRunner
        self.experiment_runner = ExperimentRunner()
        
        # Сохраняем текущую визуализацию
        self.current_figure = None
        self.current_canvas = None
        self.current_toolbar = None
        
        # Словарь для хранения всех графиков
        self.figures = {}
        
        # Создаем элементы интерфейса
        self.create_widgets()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        # Настраиваем стиль
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')  # Попробуем использовать более современную тему
        except:
            pass
        
        # Главный фрейм с панелью меню
        self.create_menu()
        
        # Создаем главный PanedWindow для разделения интерфейса
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель для настроек
        left_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(left_frame, weight=1)
        
        # Правая панель для вывода
        right_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(right_frame, weight=3)
        
        # Настройки в левой панели
        # Фрейм для выбора набора данных
        dataset_frame = ttk.LabelFrame(left_frame, text="Выбор набора данных", padding="10")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Радиокнопки для выбора встроенного набора данных
        self.dataset_var = tk.StringVar(value="iris")
        
        ttk.Label(dataset_frame, text="Встроенные наборы данных:").grid(row=0, column=0, sticky=tk.W)
        
        ttk.Radiobutton(dataset_frame, text="Iris", variable=self.dataset_var, value="iris").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="Wine", variable=self.dataset_var, value="wine").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="Breast Cancer", variable=self.dataset_var, value="breast_cancer").grid(row=3, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="MNIST (подвыборка)", variable=self.dataset_var, value="digits").grid(row=4, column=0, sticky=tk.W)
        
        # Кнопка для загрузки пользовательского набора данных
        ttk.Label(dataset_frame, text="Пользовательский набор данных:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        
        ttk.Button(dataset_frame, text="Загрузить CSV файл", command=self.load_custom_dataset).grid(row=6, column=0, sticky=tk.W, padx=5)
        
        self.custom_dataset_label = ttk.Label(dataset_frame, text="Файл не выбран")
        self.custom_dataset_label.grid(row=7, column=0, sticky=tk.W, padx=5)
        
        # Фрейм для настройки параметров шума
        noise_frame = ttk.LabelFrame(left_frame, text="Параметры шума", padding="10")
        noise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(noise_frame, text="Минимальное значение шума:").grid(row=0, column=0, sticky=tk.W)
        self.min_noise_var = tk.DoubleVar(value=0.0)
        ttk.Entry(noise_frame, textvariable=self.min_noise_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Максимальное значение шума:").grid(row=1, column=0, sticky=tk.W)
        self.max_noise_var = tk.DoubleVar(value=0.5)
        ttk.Entry(noise_frame, textvariable=self.max_noise_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Шаг изменения шума:").grid(row=2, column=0, sticky=tk.W)
        self.noise_step_var = tk.DoubleVar(value=0.1)
        ttk.Entry(noise_frame, textvariable=self.noise_step_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Количество экспериментов:").grid(row=3, column=0, sticky=tk.W)
        self.n_experiments_var = tk.IntVar(value=3)
        ttk.Entry(noise_frame, textvariable=self.n_experiments_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)

        # Флажки для выбора типов шума
        ttk.Label(noise_frame, text="Типы шума для эксперимента:").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        self.noise_types = {
            'gaussian': tk.BooleanVar(value=True),
            'uniform': tk.BooleanVar(value=True),
            'impulse': tk.BooleanVar(value=True),
            'missing': tk.BooleanVar(value=True),
            'salt_pepper': tk.BooleanVar(value=False),
            'multiplicative': tk.BooleanVar(value=False)
        }
        
        noise_type_frame = ttk.Frame(noise_frame)
        noise_type_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Checkbutton(noise_type_frame, text="Гауссовский", variable=self.noise_types['gaussian']).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(noise_type_frame, text="Равномерный", variable=self.noise_types['uniform']).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(noise_type_frame, text="Импульсный", variable=self.noise_types['impulse']).grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(noise_type_frame, text="Пропущенные значения", variable=self.noise_types['missing']).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(noise_type_frame, text="Соль и перец", variable=self.noise_types['salt_pepper']).grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(noise_type_frame, text="Мультипликативный", variable=self.noise_types['multiplicative']).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Дополнительные параметры
        additional_frame = ttk.LabelFrame(left_frame, text="Дополнительные параметры", padding="10")
        additional_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Применение предобработки
        self.use_preprocessing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(additional_frame, text="Применять предобработку данных", 
                      variable=self.use_preprocessing_var).grid(row=0, column=0, sticky=tk.W)
        
        # Сохранение моделей
        self.save_best_models_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(additional_frame, text="Сохранять лучшую модель", 
                      variable=self.save_best_models_var).grid(row=1, column=0, sticky=tk.W)
        
        # Выбор метрики для отображения
        ttk.Label(additional_frame, text="Метрика для графиков:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.metric_var = tk.StringVar(value="accuracy")
        ttk.Radiobutton(additional_frame, text="Точность", variable=self.metric_var, value="accuracy").grid(row=3, column=0, sticky=tk.W)
        ttk.Radiobutton(additional_frame, text="F1-мера", variable=self.metric_var, value="f1").grid(row=4, column=0, sticky=tk.W)
        
        # Кнопки управления
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(control_frame, text="Запустить эксперименты", 
                 command=self.run_experiments, style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Загрузить модели", 
                 command=self.load_models).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Очистить", 
                 command=self.clear_output).pack(fill=tk.X, pady=2)
        
        # Правая панель для вывода
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка для вывода текста
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="Журнал")
        
        # Текстовое поле с прокруткой
        text_scroll = ttk.Scrollbar(self.text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_output = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set)
        self.text_output.pack(fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text_output.yview)
        
        # Вкладка для визуализации результатов
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Графики")
        
        # Создаем фрейм с инструментами для графиков
        self.plot_control_frame = ttk.Frame(self.plot_frame)
        self.plot_control_frame.pack(fill=tk.X)
        
        ttk.Label(self.plot_control_frame, text="Тип шума:").pack(side=tk.LEFT, padx=5)
        self.noise_type_var = tk.StringVar()
        self.noise_type_combo = ttk.Combobox(self.plot_control_frame, textvariable=self.noise_type_var, state='readonly')
        self.noise_type_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.plot_control_frame, text="Тип графика:").pack(side=tk.LEFT, padx=5)
        self.plot_type_var = tk.StringVar(value="general")
        plot_types = [
            ("Общий график", "general"),
            ("Сравнение моделей", "compare"),
            ("Влияние предобработки", "preprocessing")
        ]
        self.plot_type_combo = ttk.Combobox(self.plot_control_frame, textvariable=self.plot_type_var, 
                                          values=[t[0] for t in plot_types], state='readonly')
        self.plot_type_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.plot_control_frame, text="Обновить график", 
                 command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.plot_control_frame, text="Сохранить график", 
                 command=self.save_current_figure).pack(side=tk.LEFT, padx=5)
        
        # Фрейм для отображения графика
        self.plot_display_frame = ttk.Frame(self.plot_frame)
        self.plot_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка для таблицы с результатами
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Таблица результатов")
        
        # Создаем фрейм с кнопками для таблицы
        self.table_control_frame = ttk.Frame(self.table_frame)
        self.table_control_frame.pack(fill=tk.X)
        
        ttk.Button(self.table_control_frame, text="Обновить таблицу", 
                 command=self.show_results_table).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.table_control_frame, text="Сохранить отчет", 
                 command=self.save_report).pack(side=tk.LEFT, padx=5)
        
        # Фрейм для отображения таблицы
        self.table_display_frame = ttk.Frame(self.table_frame)
        self.table_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка для статистики и информации
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Статистика")
        
        # Перенаправляем вывод в текстовое поле
        self.redirect_output()
    
    def create_menu(self):
        """Создает главное меню приложения"""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # Меню "Файл"
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить набор данных...", command=self.load_custom_dataset)
        file_menu.add_command(label="Загрузить модели...", command=self.load_models)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить модели...", command=self.save_models)
        file_menu.add_command(label="Сохранить отчет...", command=self.save_report)
        file_menu.add_command(label="Сохранить текущий график...", command=self.save_current_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.destroy)
        
        # Меню "Эксперимент"
        experiment_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Эксперимент", menu=experiment_menu)
        experiment_menu.add_command(label="Запустить эксперименты", command=self.run_experiments)
        experiment_menu.add_command(label="Остановить эксперимент", command=self.stop_experiment)
        experiment_menu.add_separator()
        experiment_menu.add_command(label="Очистить результаты", command=self.clear_output)
        
        # Меню "Визуализация"
        visualization_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Визуализация", menu=visualization_menu)
        visualization_menu.add_command(label="Общие графики", command=lambda: self.update_visualization(plot_type="general"))
        visualization_menu.add_command(label="Сравнение моделей", command=lambda: self.update_visualization(plot_type="compare"))
        visualization_menu.add_command(label="Влияние предобработки", command=lambda: self.update_visualization(plot_type="preprocessing"))
        
        # Меню "Справка"
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Справка", command=self.show_help)
    
    def run_experiments(self):
        """Запускает эксперименты с выбранными параметрами"""
        try:
            # Получаем параметры
            dataset = self.dataset_var.get()
            min_noise = self.min_noise_var.get()
            max_noise = self.max_noise_var.get()
            noise_step = self.noise_step_var.get()
            n_experiments = self.n_experiments_var.get()
            use_preprocessing = self.use_preprocessing_var.get()
            
            # Проверяем параметры
            if min_noise < 0:
                raise ValueError("Минимальное значение шума должно быть неотрицательным")
            
            if max_noise <= min_noise:
                raise ValueError("Максимальное значение шума должно быть больше минимального")
            
            if noise_step <= 0:
                raise ValueError("Шаг изменения шума должен быть положительным")
            
            if n_experiments <= 0:
                raise ValueError("Количество экспериментов должно быть положительным")
            
            # Переключаемся на вкладку журнала для отображения прогресса
            self.notebook.select(self.text_frame)
            
            # Загружаем набор данных
            if dataset == "custom":
                if hasattr(self, 'custom_dataset_path'):
                    self.experiment_runner.load_dataset(dataset_path=self.custom_dataset_path)
                else:
                    raise ValueError("Пользовательский набор данных не выбран")
            else:
                self.experiment_runner.load_dataset(dataset_name=dataset)
            
            # Получаем выбранные типы шума
            selected_noise_types = [name for name, var in self.noise_types.items() if var.get()]
            
            if not selected_noise_types:
                raise ValueError("Необходимо выбрать хотя бы один тип шума")
            
            # Запускаем эксперименты
            for noise_type in selected_noise_types:
                print(f"\n{'=' * 50}")
                print(f"Запуск экспериментов с шумом типа {noise_type}")
                print(f"{'=' * 50}")
                
                self.experiment_runner.run_experiment(
                    noise_type, (min_noise, max_noise), noise_step, n_experiments, use_preprocessing
                )
            
            # Обновляем выпадающий список с типами шума для визуализации
            self.update_noise_type_combobox()
            
            messagebox.showinfo("Информация", "Эксперименты успешно завершены")
            
            # Отображаем результаты
            self.show_results_table()
            self.update_visualization()
            
            # Переключаемся на вкладку с графиками
            self.notebook.select(self.plot_frame)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def stop_experiment(self):
        """Останавливает текущий эксперимент"""
        # Этот метод будет реализован в будущем
        messagebox.showinfo("Информация", "Функция остановки эксперимента пока не реализована")
    
    def update_noise_type_combobox(self):
        """Обновляет выпадающий список с типами шума на основе имеющихся результатов"""
        if hasattr(self.experiment_runner, 'experiment_results') and self.experiment_runner.experiment_results:
            noise_types = list(self.experiment_runner.experiment_results.keys())
            self.noise_type_combo['values'] = noise_types
            if noise_types:
                self.noise_type_var.set(noise_types[0])
                
    def show_about(self):
        """Показывает информацию о программе"""
        about_text = """
Программный комплекс для классификации зашумленных данных

Версия: 1.0

Данный программный комплекс предназначен для решения задачи 
классификации зашумленных данных с использованием ансамблевых
методов машинного обучения.

Разработан в рамках магистерской диссертации.
"""
        messagebox.showinfo("О программе", about_text)
    
    def show_help(self):
        """Показывает справочную информацию"""
        help_text = """
Краткая инструкция по использованию:

1. Выберите набор данных в левой панели.
2. Настройте параметры шума:
   - Минимальный и максимальный уровень шума
   - Шаг изменения уровня шума
   - Количество экспериментов для каждого уровня
3. Выберите типы шума для тестирования
4. Дополнительные параметры позволяют включить/отключить
   предобработку данных и выбрать метрики для отображения
5. Нажмите "Запустить эксперименты"
6. После завершения экспериментов вы можете:
   - Просматривать графики результатов
   - Изучать подробную таблицу с метриками
   - Сохранять модели и графики

Для сохранения графиков перейдите на вкладку "Графики",
выберите тип шума и тип графика, затем нажмите
"Сохранить график".
"""
        messagebox.showinfo("Справка", help_text)
    
    def update_visualization(self, plot_type=None):
        """Обновляет визуализацию результатов экспериментов
        
        Args:
            plot_type: Тип графика для отображения (если None, берется из комбобокса)
        """
        try:
            if not self.experiment_runner.experiment_results:
                raise ValueError("Нет результатов экспериментов для визуализации")
            
            # Определяем тип графика
            if plot_type is None:
                # Получаем выбранный тип графика из комбобокса
                selected_plot_type = self.plot_type_var.get()
                # Преобразуем название в код
                plot_types_map = {
                    "Общий график": "general",
                    "Сравнение моделей": "compare",
                    "Влияние предобработки": "preprocessing"
                }
                plot_type = plot_types_map.get(selected_plot_type, "general")
            
            # Получаем тип шума
            noise_type = self.noise_type_var.get() if self.noise_type_var.get() else None
            
            # Очищаем фрейм для отображения графика
            for widget in self.plot_display_frame.winfo_children():
                widget.destroy()
            
            # Создаем фигуру с графиками в зависимости от выбранного типа
            fig = None
            if plot_type == "general":
                # Общий график для выбранного типа шума или всех типов
                fig = self.experiment_runner.visualize_results(
                    noise_type=noise_type, 
                    metric=self.metric_var.get(),
                    figsize=(10, 6)
                )
            elif plot_type == "compare":
                # График сравнения моделей для выбранного типа шума и наихудшего уровня шума
                if noise_type:
                    fig = self.experiment_runner.visualize_single_noise_level(
                        noise_type=noise_type,
                        figsize=(10, 6)
                    )
                else:
                    raise ValueError("Для сравнения моделей необходимо выбрать тип шума")
            elif plot_type == "preprocessing":
                # График влияния предобработки
                try:
                    fig = self.experiment_runner.visualize_preprocessing_impact(figsize=(10, 6))
                except ValueError:
                    raise ValueError("Отсутствуют данные о влиянии предобработки. Необходимо запустить эксперименты с включенной предобработкой.")
            
            if fig:
                # Создаем канвас для отображения графика
                canvas = FigureCanvasTkAgg(fig, master=self.plot_display_frame)
                canvas.draw()
                
                # Добавляем панель инструментов для навигации по графику
                toolbar = NavigationToolbar2Tk(canvas, self.plot_display_frame)
                toolbar.update()
                
                # Упаковываем канвас и панель инструментов
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                # Сохраняем текущую фигуру и канвас
                self.current_figure = fig
                self.current_canvas = canvas
                self.current_toolbar = toolbar
                
                # Сохраняем фигуру в словаре для быстрого доступа
                key = f"{noise_type}_{plot_type}" if noise_type else f"all_{plot_type}"
                self.figures[key] = fig
            
            # Переключаемся на вкладку с графиками
            self.notebook.select(self.plot_frame)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при визуализации: {str(e)}")
    
    def save_current_figure(self):
        """Сохраняет текущую фигуру в файл"""
        try:
            if self.current_figure is None:
                raise ValueError("Нет графика для сохранения")
            
            # Запрашиваем имя файла для сохранения
            filetypes = [
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("JPEG", "*.jpg")
            ]
            
            filename = filedialog.asksaveasfilename(
                title="Сохранить график",
                defaultextension=".png",
                filetypes=filetypes
            )
            
            if filename:
                # Определяем формат из расширения файла
                ext = filename.split('.')[-1].lower()
                formats = [ext]
                
                # Сохраняем фигуру
                self.experiment_runner.save_figure(self.current_figure, filename.rsplit('.', 1)[0], formats)
                
                messagebox.showinfo("Информация", f"График успешно сохранен в файл {filename}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при сохранении графика: {str(e)}")
    
    def show_results_table(self):
        """Отображает таблицу с результатами экспериментов"""
        try:
            if not self.experiment_runner.experiment_results:
                raise ValueError("Нет результатов экспериментов для отображения")
            
            # Очищаем фрейм с таблицей
            for widget in self.table_display_frame.winfo_children():
                widget.destroy()
            
            # Получаем DataFrame с результатами
            report_df = self.experiment_runner.generate_report()
            
            # Создаем прокручиваемый фрейм
            table_scroll_frame = ttk.Frame(self.table_display_frame)
            table_scroll_frame.pack(fill=tk.BOTH, expand=True)
            
            # Добавляем прокрутку
            x_scroll = ttk.Scrollbar(table_scroll_frame, orient=tk.HORIZONTAL)
            y_scroll = ttk.Scrollbar(table_scroll_frame, orient=tk.VERTICAL)
            
            # Создаем таблицу
            table = ttk.Treeview(
                table_scroll_frame,
                columns=list(report_df.columns),
                show="headings",
                xscrollcommand=x_scroll.set,
                yscrollcommand=y_scroll.set
            )
            
            # Настраиваем прокрутку
            x_scroll.config(command=table.xview)
            y_scroll.config(command=table.yview)
            
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Задаем заголовки столбцов и регулируем ширину
            for column in report_df.columns:
                table.heading(column, text=column)
                
                # Устанавливаем ширину столбца в зависимости от содержимого
                if column in ['Тип шума', 'Уровень шума']:
                    table.column(column, width=100, anchor=tk.CENTER)
                elif column in ['Эффект предобработки']:
                    table.column(column, width=150, anchor=tk.CENTER)
                else:
                    table.column(column, width=120, anchor=tk.CENTER)
            
            # Заполняем таблицу данными
            for i, row in report_df.iterrows():
                # Цветовое выделение строк для лучшей читаемости
                if i % 2 == 0:
                    table.insert("", tk.END, values=list(row), tags=('evenrow',))
                else:
                    table.insert("", tk.END, values=list(row), tags=('oddrow',))
            
            # Создаем теги для оформления строк
            table.tag_configure('evenrow', background='#f0f0f0')
            table.tag_configure('oddrow', background='#ffffff')
            
            # Сохраняем DataFrame для последующего использования
            self.report_df = report_df
            
            # Переключаемся на вкладку с таблицей
            self.notebook.select(self.table_frame)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при отображении таблицы: {str(e)}")
    
    def save_models(self):
        """Сохраняет обученные модели"""
        try:
            if not hasattr(self.experiment_runner.model_builder, 'models') or not self.experiment_runner.model_builder.models:
                raise ValueError("Нет обученных моделей для сохранения")
            
            # Запрашиваем директорию для сохранения
            save_dir = filedialog.askdirectory(title="Выберите директорию для сохранения моделей")
            
            if save_dir:
                self.experiment_runner.save_models(path=save_dir)
                messagebox.showinfo("Информация", f"Модели успешно сохранены в директории {save_dir}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при сохранении моделей: {str(e)}")
    
    def load_models(self):
        """Загружает обученные модели"""
        try:
            # Запрашиваем директорию с моделями
            load_dir = filedialog.askdirectory(title="Выберите директорию с сохраненными моделями")
            
            if load_dir:
                models = self.experiment_runner.load_models(path=load_dir)
                
                if models:
                    # После загрузки моделей можно сразу проверить их на каком-либо тестовом наборе
                    messagebox.showinfo("Информация", f"Модели успешно загружены. Загружено {len(models)} моделей.")
                    
                    # Выводим информацию о загруженных моделях
                    print("\nЗагруженные модели:")
                    for name in models.keys():
                        print(f"  - {name}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при загрузке моделей: {str(e)}")
    
    def save_report(self):
        """Сохраняет отчет о результатах экспериментов"""
        try:
            # Сначала обновим таблицу результатов
            if not hasattr(self, 'report_df') or self.report_df is None:
                self.report_df = self.experiment_runner.generate_report()
            
            if self.report_df is None or self.report_df.empty:
                raise ValueError("Нет данных для сохранения отчета")
            
            # Запрашиваем имя файла для сохранения
            file_path = filedialog.asksaveasfilename(
                title="Сохранить отчет",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Сохраняем отчет
                if file_path.endswith('.xlsx'):
                    self.report_df.to_excel(file_path, index=False)
                    print(f"Отчет сохранен в формате Excel: {file_path}")
                elif file_path.endswith('.csv'):
                    self.report_df.to_csv(file_path, index=False)
                    print(f"Отчет сохранен в формате CSV: {file_path}")
                else:
                    self.report_df.to_excel(file_path, index=False)
                    print(f"Отчет сохранен в формате Excel: {file_path}")
                
                messagebox.showinfo("Информация", f"Отчет успешно сохранен в файле {file_path}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка при сохранении отчета: {str(e)}")
    
    def clear_output(self):
        """Очищает вывод и сбрасывает данные эксперимента"""
        # Запрашиваем подтверждение
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить все результаты? Все несохраненные данные будут потеряны."):
            # Очищаем текстовое поле
            self.text_output.delete(1.0, tk.END)
            
            # Очищаем графики
            for widget in self.plot_display_frame.winfo_children():
                widget.destroy()
            
            # Очищаем таблицу
            for widget in self.table_display_frame.winfo_children():
                widget.destroy()
            
            # Сбрасываем текущую фигуру и канвас
            self.current_figure = None
            self.current_canvas = None
            self.current_toolbar = None
            
            # Очищаем словарь фигур
            self.figures = {}
            
            # Сбрасываем данные эксперимента
            self.experiment_runner = ExperimentRunner()
            
            # Очищаем комбобокс с типами шума
            self.noise_type_combo['values'] = []
            self.noise_type_var.set('')
            
            if hasattr(self, 'report_df'):
                del self.report_df
            
            print("Вывод очищен. Готов к новым экспериментам.")
    
    def load_custom_dataset(self):
        """Загружает пользовательский набор данных"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Проверяем, что файл существует и может быть прочитан
                test_df = pd.read_csv(file_path, nrows=5)
                n_columns = test_df.shape[1]
                
                if n_columns < 2:
                    raise ValueError(f"Файл должен содержать как минимум 2 столбца (признаки и метка класса), найдено {n_columns}")
                
                self.custom_dataset_label.config(text=f"Выбран файл: {os.path.basename(file_path)}")
                self.dataset_var.set("custom")
                self.custom_dataset_path = file_path
                print(f"Выбран пользовательский набор данных: {file_path}")
                print(f"Обнаружено {n_columns} столбцов. Первые 5 строк:")
                print(test_df.head())
                
                # Переключаемся на вкладку с журналом для просмотра информации
                self.notebook.select(self.text_frame)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {str(e)}")
                print(f"Ошибка при загрузке файла: {str(e)}")
                self.custom_dataset_label.config(text="Файл не выбран")
    
    def redirect_output(self):
        """Перенаправляет стандартный вывод в текстовое поле"""
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                self.buffer = ""
            
            def write(self, string):
                self.buffer += string
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
            
            def flush(self):
                if self.buffer:
                    self.buffer = ""
        
        import sys
        sys.stdout = TextRedirector(self.text_output)


# Запуск приложения
if __name__ == "__main__":
    # Настройка стиля Tkinter
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")  # Используем современную тему, если доступна
    except ImportError:
        root = tk.Tk()
        print("Пакет 'ttkthemes' не установлен. Используется стандартная тема.")
    
    # Запускаем приложение
    app = NoisyDataClassificationApp(root)
    
    # Устанавливаем обработчик закрытия окна
    def on_closing():
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Запускаем главный цикл приложения
    root.mainloop()