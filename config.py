# ⚙️ КОНФИГУРАЦИОННЫЙ ФАЙЛ

import os
from datetime import timedelta

# ========================
# ОСНОВНАЯ КОНФИГУРАЦИЯ
# ========================

# Секретный ключ для сессий
SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'

# Database configuration
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///keystroke_auth.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Session configuration
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
SESSION_TIMEOUT = 3600  # 1 час в секундах

# ========================
# ПАРАМЕТРЫ АУТЕНТИФИКАЦИИ
# ========================


class AuthConfig:
    """Параметры биометрической аутентификации"""
    
    # Энролмент
    ENROLLMENT_SAMPLES = 5                                           # Минимум образцов для энролмента
    ENROLLMENT_TEXT = "The quick brown fox jumps over the lazy dog"  # Текст для энролмента

    # Верификация
    VERIFICATION_THRESHOLD = 0.6          # Порог для верификации (0-1)
    CONTINUOUS_AUTH_THRESHOLD = 0.5       # Порог для непрерывной проверки
    VERIFY_THRESHOLD = 0.60
    VERIFICATION_TEXT = "The quick brown fox jumps over the lazy dog"

    # Аномалии
    MAX_ANOMALY_WARNINGS = 3              # Максимум предупреждений перед блокировкой
    ANOMALY_CHECK_INTERVAL = 20           # Проверка каждые N символов
    ANOMALY_DETECTOR_CONTAMINATION = 0.1  # Expected anomaly rate for Isolation Forest

    # Таймауты
    SESSION_TIMEOUT_SECONDS = 3600        # Таймаут сессии (1 час)
    INACTIVITY_CHECK_INTERVAL = 60        # Проверка активности каждые N сек

    # Настройки алгоритма
    EUCLIDEAN_DISTANCE_THRESHOLD = 5.0    # Для отладки
    FEATURE_VECTOR_SIZE = 10              # Размер вектора признаков


# ========================
# ПАРАМЕТРЫ ПРИЛОЖЕНИЯ
# ========================

class AppConfig:
    """Основные параметры приложения"""
    
    # Flask
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    TESTING = False
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    
    # Пагинация
    ITEMS_PER_PAGE = 20
    
    # Логирование
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ========================
# ПАРАМЕТРЫ РАЗРАБОТКИ
# ========================

class DevelopmentConfig:
    """Конфигурация для разработки"""
    
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Выводить SQL запросы
    TESTING = False
    SECRET_KEY = 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///keystroke_auth_dev.db'
    SESSION_TIMEOUT = 7200  # 2 часа для удобства разработки


# ========================
# ПАРАМЕТРЫ ТЕСТИРОВАНИЯ
# ========================

class TestingConfig:
    """Конфигурация для тестирования"""
    
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test-secret-key'
    WTF_CSRF_ENABLED = False
    SESSION_TIMEOUT = 300  # 5 минут


# ========================
# ВЫБОР КОНФИГУРАЦИИ
# ========================

def get_config(env='development'):
    """Получить конфигурацию на основе окружения"""
    
    config_map = {
        'development': DevelopmentConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)


# ========================
# ЛОГИРОВАНИЕ
# ========================

import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'keystroke_auth.log',
            'formatter': 'detailed',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        'keystroke_app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    }
}


# ========================
# ИНИЦИАЛИЗАЦИЯ ЛОГИРОВАНИЯ
# ========================

def init_logging():
    """Инициализировать логирование"""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('keystroke_app')
    return logger
