# 🔧 Инструкция по интеграции дополнительных маршрутов

## Как добавить дополнительные маршруты в keystroke_app.py

### Шаг 1: Найдите место для добавления

В файле `keystroke_app.py` найдите строку:
```python
@app.route('/api/profile-stats')
@login_required
def profile_stats():
```

### Шаг 2: Добавьте новые маршруты перед `if __name__ == '__main__':`

Скопируйте код из `ROUTES_ADDON.py` и вставьте его перед строкой:
```python
if __name__ == '__main__':
```

### Шаг 3: Проверьте импорты

Убедитесь, что в начале файла есть:
```python
from sklearn.metrics import roc_auc_score, auc
```

Если это не так, добавьте эту строку после других импортов scikit-learn.

## Новые API эндпоинты

### 1. Верификация клавиатурного почерка
```
GET /verify-keystroke
```
Страница верификации почерка при входе.

### 2. Экспорт данных
```
GET /api/export-data
Authorization: Требуется
```
Экспортирует все события нажатия клавиш пользователя в JSON формате.

**Ответ:**
```json
[
  {
    "timestamp": "2024-01-01T12:00:00",
    "key_char": "A",
    "keycode": 65,
    "press_time": 100,
    "release_time": 150,
    "dwell_time": 50
  }
]
```

### 3. Информация о сессии
```
GET /api/session-info
Authorization: Требуется
```
Получить информацию о текущей активной сессии.

**Ответ:**
```json
{
  "user_id": 1,
  "login_time": "2024-01-01T12:00:00",
  "last_activity": "2024-01-01T12:05:30",
  "authentication_score": 0.85,
  "is_active": true
}
```

### 4. Вычисление метрик
```
POST /api/calculate-metrics
Authorization: Требуется
Content-Type: application/json
```

**Запрос:**
```json
{
  "test_scores": [
    [1, 0.92],  // [label: 1=genuine, 0=impostor, score: 0-1]
    [1, 0.88],
    [1, 0.95],
    [0, 0.45],
    [0, 0.35],
    [0, 0.52]
  ]
}
```

**Ответ:**
```json
{
  "genuine_scores": {
    "mean": 0.917,
    "std": 0.035,
    "min": 0.88,
    "max": 0.95
  },
  "impostor_scores": {
    "mean": 0.44,
    "std": 0.085,
    "min": 0.35,
    "max": 0.52
  },
  "eer": 0.0583,
  "eer_threshold": 0.685,
  "auc": 0.964,
  "accuracy_at_eer": 0.9417
}
```

## Использование для научных работ

### Пример: Вычисление метрик для статьи

```python
import requests

# Результаты тестирования (label: 1=genuine, 0=impostor)
test_data = [
    [1, 0.92], [1, 0.88], [1, 0.95],  # genuine scores
    [0, 0.45], [0, 0.35], [0, 0.52]   # impostor scores
]

# Отправить запрос
response = requests.post(
    'http://localhost:5000/api/calculate-metrics',
    json={'test_scores': test_data},
    cookies={'session': 'YOUR_SESSION_ID'}
)

metrics = response.json()
print(f"EER: {metrics['eer']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy at EER: {metrics['accuracy_at_eer']:.4f}")
```

### Пример: Экспорт данных для анализа

```python
import pandas as pd

# Получить данные
response = requests.get(
    'http://localhost:5000/api/export-data',
    cookies={'session': 'YOUR_SESSION_ID'}
)

data = response.json()

# Преобразовать в DataFrame
df = pd.DataFrame(data)

# Анализ
print(df.describe())

# Сохранить в CSV
df.to_csv('keystroke_data.csv', index=False)

# Статистика
print(f"Среднее dwell time: {df['dwell_time'].mean():.2f} мс")
print(f"Среднее typing speed: {len(data) / (data[-1]['release_time'] / 1000):.2f} сим/сек")
```

## Структура данных в БД

### keystroke_events таблица
```sql
CREATE TABLE keystroke_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY,
    session_id TEXT,
    keycode INTEGER,
    key_char TEXT,
    press_time FLOAT,
    release_time FLOAT,
    timestamp DATETIME
)
```

### keystroke_profiles таблица
```sql
CREATE TABLE keystroke_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY,
    features TEXT,  -- JSON с extracted features
    created_at DATETIME,
    sample_count INTEGER
)
```

## Получение SQL запросов для анализа

### Все события пользователя
```sql
SELECT 
    ke.timestamp,
    ke.key_char,
    ke.keycode,
    ke.press_time,
    ke.release_time,
    (ke.release_time - ke.press_time) as dwell_time
FROM keystroke_events ke
JOIN users u ON ke.user_id = u.id
WHERE u.username = 'YOUR_USERNAME'
ORDER BY ke.timestamp;
```

### Статистика по пользователям
```sql
SELECT 
    u.username,
    COUNT(kp.id) as profile_count,
    COUNT(ke.id) as keystroke_events,
    AVG(ke.release_time - ke.press_time) as avg_dwell_time
FROM users u
LEFT JOIN keystroke_profiles kp ON u.id = kp.user_id
LEFT JOIN keystroke_events ke ON u.id = ke.user_id
GROUP BY u.username;
```

## Советы для использования в PyCharm

### Удаленное отладка приложения

1. Откройте PyCharm
2. Нажмите "Edit Configurations"
3. Добавьте новую конфигурацию Python
4. Script path: `/path/to/keystroke_app.py`
5. Запустите с отладкой (Shift+F9)

### Интеграция с IDE

```python
# Добавьте в keystroke_app.py для отладки
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, port=5000, use_reloader=False)
```

## Проверка работоспособности

### Тест 1: Регистрация и энролмент
```bash
curl -X POST http://localhost:5000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@test.com","password":"pass123"}'
```

### Тест 2: Вычисление метрик
```bash
curl -X POST http://localhost:5000/api/calculate-metrics \
  -H "Content-Type: application/json" \
  -d '{"test_scores":[[1,0.9],[1,0.85],[0,0.4],[0,0.35]]}'
```

## Производительность и оптимизация

### Для больших датасетов

1. **Кэширование профилей**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_user_profile(user_id):
    return KeystrokeProfile.query.filter_by(user_id=user_id).first()
```

2. **Асинхронная обработка**
```python
from celery import Celery

celery_app = Celery('keystroke_app', broker='redis://localhost:6379')

@celery_app.task
def analyze_keystroke_batch(events):
    # Долгая операция анализа
    pass
```

3. **Индексирование БД**
```sql
CREATE INDEX idx_keystroke_user ON keystroke_events(user_id);
CREATE INDEX idx_keystroke_timestamp ON keystroke_events(timestamp);
```

## Заключение

Дополнительные маршруты расширяют функциональность приложения для:
✓ Научных исследований
✓ Экспорта и анализа данных
✓ Вычисления биометрических метрик
✓ Интеграции с другими системами
✓ Мониторинга сессий

Для вопросов смотрите файлы README.md, SCIENTIFIC.md и INSTALL.md.
