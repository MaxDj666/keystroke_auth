# ⚙️ КОНФИГУРАЦИОННЫЙ ФАЙЛ

# Копируйте этот файл как config.py и используйте в keystroke_app.py

import os
from datetime import timedelta

# ========================
# ОСНОВНАЯ КОНФИГУРАЦИЯ
# ========================

# Секретный ключ для сессий (измените на production)
SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

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
    ENROLLMENT_SAMPLES = 5              # Минимум образцов для энролмента
    ENROLLMENT_TEXT = "The quick brown fox jumps over the lazy dog"  # Текст для энролмента
    
    # Верификация
    VERIFICATION_THRESHOLD = 0.6        # Порог для верификации (0-1)
    CONTINUOUS_AUTH_THRESHOLD = 0.5    # Порог для непрерывной проверки
    
    # Аномалии
    MAX_ANOMALY_WARNINGS = 3            # Максимум предупреждений перед блокировкой
    ANOMALY_CHECK_INTERVAL = 20         # Проверка каждые N символов
    ANOMALY_DETECTOR_CONTAMINATION = 0.1  # Expected anomaly rate for Isolation Forest
    
    # Таймауты
    SESSION_TIMEOUT_SECONDS = 3600      # Таймаут сессии (1 час)
    INACTIVITY_CHECK_INTERVAL = 60      # Проверка активности каждые N сек
    
    # Настройки алгоритма
    EUCLIDEAN_DISTANCE_THRESHOLD = 5.0  # Для отладки
    FEATURE_VECTOR_SIZE = 10            # Размер вектора признаков


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
    SECRET_KEY = 'dev-secret-key-insecure'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///keystroke_auth_dev.db'
    SESSION_TIMEOUT = 7200  # 2 часа для удобства разработки


# ========================
# ПАРАМЕТРЫ PRODUCTION
# ========================

class ProductionConfig:
    """Конфигурация для production"""
    
    DEBUG = False
    SQLALCHEMY_ECHO = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SESSION_TIMEOUT = 1800  # 30 минут
    
    # Требуется HTTPS
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


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
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)


# ========================
# КАК ИСПОЛЬЗОВАТЬ
# ========================

"""
# В keystroke_app.py:

from config import AuthConfig, get_config

# Получить конфигурацию
env = os.environ.get('FLASK_ENV', 'development')
config = get_config(env)

# Применить к приложению
app.config.from_object(config)

# Использовать параметры аутентификации
analyzer = KeystrokeDynamicsAnalyzer()
analyzer.THRESHOLD = AuthConfig.VERIFICATION_THRESHOLD
analyzer.ENROLLMENT_SAMPLES = AuthConfig.ENROLLMENT_SAMPLES
"""


# ========================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (.env)
# ========================

"""
Создайте файл .env с переменными окружения:

FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=postgresql://user:password@localhost/keystroke_db
DEBUG=False
LOG_LEVEL=INFO
"""


# ========================
# ПРИМЕРЫ ПАРАМЕТРИЗАЦИИ
# ========================

class ExperimentConfigs:
    """Предустановки для различных экспериментов"""
    
    # Высокая безопасность
    HIGH_SECURITY = {
        'THRESHOLD': 0.8,
        'CONTINUOUS_AUTH_THRESHOLD': 0.7,
        'MAX_ANOMALY_WARNINGS': 1,
        'ANOMALY_CHECK_INTERVAL': 10
    }
    
    # Удобство пользователя
    HIGH_USABILITY = {
        'THRESHOLD': 0.5,
        'CONTINUOUS_AUTH_THRESHOLD': 0.4,
        'MAX_ANOMALY_WARNINGS': 5,
        'ANOMALY_CHECK_INTERVAL': 30
    }
    
    # Баланс
    BALANCED = {
        'THRESHOLD': 0.6,
        'CONTINUOUS_AUTH_THRESHOLD': 0.5,
        'MAX_ANOMALY_WARNINGS': 3,
        'ANOMALY_CHECK_INTERVAL': 20
    }
    
    # Научные исследования (максимальная точность)
    RESEARCH = {
        'THRESHOLD': 0.65,
        'CONTINUOUS_AUTH_THRESHOLD': 0.55,
        'MAX_ANOMALY_WARNINGS': 2,
        'ANOMALY_CHECK_INTERVAL': 15,
        'ENROLLMENT_SAMPLES': 10,
        'ANOMALY_DETECTOR_CONTAMINATION': 0.05
    }


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
