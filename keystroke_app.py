"""
Keystroke Dynamics Authentication System
Биометрическая аутентификация на основе анализа клавиатурного почерка
"""
import datetime
import json
import os
from functools import wraps

import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from werkzeug.security import generate_password_hash, check_password_hash

# Инициализация Flask приложения
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///keystroke_auth.db'
app.config['SESSION_TIMEOUT'] = 3600  # 1 час

db = SQLAlchemy(app)
CORS(app)

# ========================
# МОДЕЛИ БАЗЫ ДАННЫХ
# ========================

class User(db.Model):
    """Модель пользователя"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))
    keystroke_profiles = db.relationship('KeystrokeProfile', backref='user', lazy=True)
    sessions = db.relationship('UserSession', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class KeystrokeProfile(db.Model):
    """Модель профиля клавиатурного почерка"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    features = db.Column(db.Text, nullable=False)  # JSON с признаками
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))
    sample_count = db.Column(db.Integer, default=1)


class KeystrokeEvent(db.Model):
    """Модель события нажатия клавиши"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    keycode = db.Column(db.Integer, nullable=False)
    key_char = db.Column(db.String(10), nullable=False)
    press_time = db.Column(db.Float, nullable=False)
    release_time = db.Column(db.Float, nullable=True)  # Разрешено NULL
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))


class UserSession(db.Model):
    """Модель сессии пользователя"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_token = db.Column(db.String(200), unique=True, nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))
    last_activity = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))
    is_active = db.Column(db.Boolean, default=True)
    authentication_score = db.Column(db.Float, default=1.0)


# ========================
# КЛАСС АНАЛИЗА ПОЧЕРКА
# ========================

class KeystrokeDynamicsAnalyzer:
    """Анализатор динамики клавиатурного почерка"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        # Пороги для аутентификации
        self.ENROLLMENT_SAMPLES = 5  # Минимум образцов для энролмента
        self.THRESHOLD = 0.6  # Порог идентификации (0-1)
        self.CONTINUOUS_AUTH_THRESHOLD = 0.5

    def extract_features(self, keystroke_events: list) -> dict:
        """Извлечение признаков из событий нажатия клавиш"""
        if len(keystroke_events) < 2:
            return None

        features = {
            'dwell_times': [],
            'flight_times': [],
            'typing_speed': 0,
            'keystroke_rhythm': [],
            'pressure_consistency': 0
        }

        # Вычисление dwell time (время держания клавиши)
        for event in keystroke_events:
            release_time = event.get('release_time')
            press_time = event.get('press_time')

            if release_time is not None and press_time is not None:
                dwell_time = release_time - press_time
                if dwell_time > 0:
                    features['dwell_times'].append(dwell_time)

        # Вычисление flight time (время между клавишами)
        for i in range(len(keystroke_events) - 1):
            current_release = keystroke_events[i].get('release_time')
            next_press = keystroke_events[i + 1].get('press_time')

            if current_release is not None and next_press is not None:
                flight_time = next_press - current_release
                if flight_time >= 0:  # Исключаем перекрытия
                    features['flight_times'].append(flight_time)

        # Вычисление скорости печати (символы в секунду)
        # Находим первое и последнее события с известными временами
        valid_events = [e for e in keystroke_events
                        if e.get('press_time') is not None and e.get('release_time') is not None]

        if len(valid_events) >= 2:
            first_event = valid_events[0]
            last_event = valid_events[-1]
            total_time = last_event['release_time'] - first_event['press_time']
            if total_time > 0:
                features['typing_speed'] = len(valid_events) / (total_time / 1000)

        # Ритм печати
        if features['dwell_times']:
            features['keystroke_rhythm'] = features['dwell_times'][:10]

        # Консистентность нажатия (вариация dwell times)
        if features['dwell_times']:
            features['pressure_consistency'] = np.std(features['dwell_times']) if len(
                features['dwell_times']) > 1 else 0

        return features

    def features_to_vector(self, features: dict) -> np.ndarray:
        """Преобразование признаков в вектор для машинного обучения"""
        if not features:
            return None

        vector = []

        # Статистики dwell times
        if features['dwell_times']:
            dwell = np.array(features['dwell_times'])
            vector.extend([
                np.mean(dwell),
                np.std(dwell),
                np.min(dwell),
                np.max(dwell)
            ])
        else:
            vector.extend([0, 0, 0, 0])

        # Статистики flight times
        if features['flight_times']:
            flight = np.array(features['flight_times'])
            vector.extend([
                np.mean(flight),
                np.std(flight),
                np.min(flight),
                np.max(flight)
            ])
        else:
            vector.extend([0, 0, 0, 0])

        # Скорость печати
        vector.append(features['typing_speed'])

        # Консистентность
        vector.append(features['pressure_consistency'])

        return np.array(vector).reshape(1, -1)

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Вычисление схожести между двумя векторами признаков"""
        if vector1 is None or vector2 is None:
            return 0.0

        # Нормализация
        combined = np.vstack([vector1, vector2])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined)

        # Евклидово расстояние
        distance = np.linalg.norm(scaled[0] - scaled[1])

        # Преобразование расстояния в подобие (обратная функция)
        similarity = 1 / (1 + distance)

        return float(similarity)

    def train_profile(self, user_vectors: list) -> dict:
        """Обучение профиля на основе нескольких образцов"""
        if len(user_vectors) < self.ENROLLMENT_SAMPLES:
            return None

        user_array = np.vstack(user_vectors)

        profile = {
            'mean_vector': np.mean(user_array, axis=0).tolist(),
            'std_vector': np.std(user_array, axis=0).tolist(),
            'sample_count': len(user_vectors),
            'created_at': datetime.datetime.now(datetime.UTC).isoformat()
        }

        return profile

    def verify_keystroke(self, keystroke_vector: np.ndarray, profile: dict) -> dict:
        """Верификация клавиатурного почерка против профиля"""
        if keystroke_vector is None or not profile:
            return {'authenticated': False, 'score': 0.0}

        mean_vector = np.array(profile['mean_vector']).reshape(1, -1)
        similarity = self.calculate_similarity(keystroke_vector, mean_vector)

        # Проверка аномалии с помощью Isolation Forest
        scaler = StandardScaler()
        combined = np.vstack([keystroke_vector, mean_vector])
        scaled = scaler.fit_transform(combined)

        is_anomaly = self.anomaly_detector.fit_predict(scaled)[0] == -1

        authenticated = similarity > self.THRESHOLD and not is_anomaly

        return {
            'authenticated': authenticated,
            'score': float(similarity),
            'is_anomaly': bool(is_anomaly)
        }

    def compare_profiles(self, features1: dict, features2: dict) -> float:
        """
        Сравнивает два профиля признаков и возвращает оценку схожести (0-1)

        Args:
            features1: Первый словарь с признаками
            features2: Второй словарь с признаками

        Returns:
            float: Оценка схожести от 0 до 1
        """
        if not features1 or not features2:
            return 0.0

        # Преобразуем признаки в векторы
        vector1 = self.features_to_vector(features1)
        vector2 = self.features_to_vector(features2)

        # Если не удалось создать векторы
        if vector1 is None or vector2 is None:
            return 0.0

        # Используем существующий метод calculate_similarity
        return self.calculate_similarity(vector1, vector2)


# Инициализация анализатора
analyzer = KeystrokeDynamicsAnalyzer()


# ========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ========================

def login_required(f):
    """Декоратор для проверки аутентификации"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def check_session_validity(user_id):
    """Проверка валидности сессии"""
    active_session = UserSession.query.filter_by(
        user_id=user_id,
        is_active=True
    ).first()

    if not active_session:
        return False

    # Проверка таймаута
    if datetime.datetime.now(datetime.UTC) - active_session.last_activity > datetime.timedelta(seconds=app.config['SESSION_TIMEOUT']):
        active_session.is_active = False
        db.session.commit()
        return False

    # Обновление времени активности
    active_session.last_activity = datetime.datetime.now(datetime.UTC)
    db.session.commit()

    return True


# ========================
# МАРШРУТЫ
# ========================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Регистрация нового пользователя"""
    if request.method == 'GET':
        return render_template('register.html')

    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Пользователь уже существует'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'Регистрация успешна. Перейдите на страницу входа.'}), 201


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Вход в приложение"""
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Неверные учетные данные'}), 401

    # Проверка наличия профиля почерка
    profiles = KeystrokeProfile.query.filter_by(user_id=user.id).all()

    if len(profiles) < analyzer.ENROLLMENT_SAMPLES:
        # Требуется энролмент
        session['temp_user_id'] = user.id
        session['temp_username'] = username
        return jsonify({
            'requires_enrollment': True,
            'message': 'Требуется регистрация клавиатурного почерка'
        }), 200

    return jsonify({
        'requires_enrollment': False,
        'message': 'Перейдите к верификации почерка'
    }), 200


@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    """Энролмент клавиатурного почерка"""
    if request.method == 'GET':
        if 'temp_user_id' not in session:
            return redirect(url_for('login'))
        return render_template('enroll.html')

    data = request.get_json()
    keystroke_events = data.get('keystroke_events', [])

    if 'temp_user_id' not in session:
        return jsonify({'error': 'Сессия истекла'}), 401

    user_id = session['temp_user_id']
    existing_profiles = KeystrokeProfile.query.filter_by(user_id=user_id).count()

    # Извлечение и анализ признаков
    features = analyzer.extract_features(keystroke_events)
    feature_vector = analyzer.features_to_vector(features)

    if feature_vector is None:
        return jsonify({'error': 'Не удалось обработать события клавиш'}), 400

    # Сохранение образца
    profile = KeystrokeProfile(
        user_id=user_id,
        features=json.dumps(features),
        sample_count=existing_profiles + 1
    )
    db.session.add(profile)
    db.session.commit()

    # Сохранение событий для анализа
    for event in keystroke_events:
        keystroke_event = KeystrokeEvent(
            user_id=user_id,
            session_id=f"enroll_{existing_profiles + 1}",
            keycode=event.get('keycode', 0),
            key_char=event.get('key_char', ''),
            press_time=event.get('press_time', 0),
            release_time=event.get('release_time', 0)
        )
        db.session.add(keystroke_event)

    db.session.commit()

    samples_needed = analyzer.ENROLLMENT_SAMPLES - existing_profiles - 1

    if samples_needed > 0:
        return jsonify({
            'success': True,
            'samples_needed': samples_needed,
            'message': f'Образец {existing_profiles + 1} записан. Требуется еще {samples_needed} образцов.'
        }), 200
    else:
        # Энролмент завершен, создаем сессию
        session.clear()
        user = User.query.get(user_id)
        session['user_id'] = user_id
        session['username'] = user.username

        # Создание записи сессии
        user_session = UserSession(
            user_id=user_id,
            session_token=os.urandom(32).hex()
        )
        db.session.add(user_session)
        db.session.commit()

        return jsonify({
            'success': True,
            'enrollment_complete': True,
            'message': 'Энролмент завершен! Добро пожаловать!'
        }), 200


@app.route('/verify-keystroke', methods=['GET', 'POST'])
def verify_keystroke():
    """Верификация почерка пользователя при входе"""
    # Разрешаем доступ как с temp_user_id (после логина), так и с user_id
    if 'temp_user_id' not in session and 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        # Загружаем страницу верификации
        return render_template('verify.html')

    else:  # POST запрос для верификации
        try:
            data = request.json
            keystroke_events = data.get('keystroke_events', [])

            # Определяем user_id (из временной или постоянной сессии)
            if 'temp_user_id' in session:
                user_id = session['temp_user_id']
                is_temp_session = True
            else:
                user_id = session['user_id']
                is_temp_session = False

            if not keystroke_events or len(keystroke_events) < 5:
                return jsonify({
                    'verified': False,
                    'message': 'Недостаточно данных для верификации'
                }), 400

            # Получаем сохраненные профили пользователя
            profiles = KeystrokeProfile.query.filter_by(
                user_id=user_id
            ).order_by(KeystrokeProfile.created_at.desc()).limit(5).all()

            if not profiles:
                return jsonify({
                    'verified': False,
                    'message': 'Профиль пользователя не найден. Пройдите энролмент.'
                }), 401

            # Анализируем текущий ввод
            current_features = analyzer.extract_features(keystroke_events)

            if not current_features:
                return jsonify({
                    'verified': False,
                    'message': 'Ошибка при анализе почерка'
                }), 400

            # Сравниваем с каждым сохраненным профилем
            max_similarity = 0

            for profile in profiles:
                stored_features = json.loads(profile.features)

                if stored_features:
                    similarity = analyzer.compare_profiles(stored_features, current_features)
                    max_similarity = max(max_similarity, similarity)

            # Пороговое значение
            VERIFY_THRESHOLD = 0.60
            verified = max_similarity >= VERIFY_THRESHOLD

            if verified:
                # Если это временная сессия (после логина), делаем ее постоянной
                if is_temp_session:
                    session['user_id'] = user_id
                    user = User.query.get(user_id)
                    session['username'] = user.username
                    # Удаляем временные данные
                    session.pop('temp_user_id', None)
                    session.pop('temp_username', None)

                    # Создание записи сессии
                    user_session = UserSession(
                        user_id=user_id,
                        session_token=os.urandom(32).hex()
                    )
                    db.session.add(user_session)
                    db.session.commit()

                return jsonify({
                    'verified': True,
                    'message': f'✅ Верификация успешна! ({max_similarity * 100:.0f}%)',
                    'similarity': max_similarity,
                    'redirect': '/dashboard'  # Указываем куда перенаправить
                })
            else:
                return jsonify({
                    'verified': False,
                    'message': f'❌ Верификация не пройдена ({max_similarity * 100:.0f}%). Попробуйте снова.',
                    'similarity': max_similarity
                }), 401

        except Exception as e:
            print(f"Error in verify_keystroke: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'verified': False
            }), 500


@app.route('/dashboard')
@login_required
def dashboard():
    """Защищенная область - мониторинг почерка"""
    if not check_session_validity(session['user_id']):
        session.clear()
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', username=user.username)


@app.route('/api/keystroke-event', methods=['POST'])
@login_required
def save_keystroke_event():
    """API для сохранения события клавиши во время работы"""
    if not check_session_validity(session['user_id']):
        return jsonify({'error': 'Сессия истекла'}), 401

    data = request.get_json()
    user_id = session['user_id']

    # Получение активной сессии
    active_session = UserSession.query.filter_by(
        user_id=user_id,
        is_active=True
    ).first()

    if not active_session:
        return jsonify({'error': 'Активная сессия не найдена'}), 401

    # Сохранение события
    keystroke_event = KeystrokeEvent(
        user_id=user_id,
        session_id=active_session.session_token,
        keycode=data.get('keycode', 0),
        key_char=data.get('key_char', ''),
        press_time=data.get('press_time', 0),
        release_time=data.get('release_time', 0)
    )
    db.session.add(keystroke_event)
    db.session.commit()

    return jsonify({'success': True}), 200


@app.route('/api/continuous-auth', methods=['POST'])
@login_required
def continuous_authentication():
    """Непрерывная аутентификация - анализ текущего почерка"""
    if not check_session_validity(session['user_id']):
        return jsonify({'authenticated': False, 'error': 'Сессия истекла'}), 401

    data = request.get_json()
    keystroke_events = data.get('keystroke_events', [])
    user_id = session['user_id']

    if len(keystroke_events) < 5:
        return jsonify({'authenticated': True, 'score': 1.0, 'message': 'Недостаточно данных'}), 200

    # Получение профиля
    profiles = KeystrokeProfile.query.filter_by(user_id=user_id).all()

    if not profiles:
        return jsonify({'authenticated': False, 'error': 'Профиль не найден'}), 404

    # Анализ текущего почерка
    features = analyzer.extract_features(keystroke_events)
    current_vector = analyzer.features_to_vector(features)

    if current_vector is None:
        return jsonify({'authenticated': True, 'score': 1.0}), 200

    # Загрузка и верификация против профиля
    all_vectors = []
    for profile in profiles:
        profile_features = json.loads(profile.features)
        vector = analyzer.features_to_vector(profile_features)
        if vector is not None:
            all_vectors.append(vector)

    trained_profile = analyzer.train_profile(all_vectors)
    result = analyzer.verify_keystroke(current_vector, trained_profile)

    # Обновление статуса сессии при низком счете
    if result['score'] < analyzer.CONTINUOUS_AUTH_THRESHOLD:
        active_session = UserSession.query.filter_by(
            user_id=user_id,
            is_active=True
        ).first()

        if active_session:
            active_session.is_active = False
            db.session.commit()

            return jsonify({
                'authenticated': False,
                'score': result['score'],
                'message': 'Обнаружено несоответствие почерка. Сессия завершена.'
            }), 401

    return jsonify({
        'authenticated': True,
        'score': result['score'],
        'is_anomaly': result['is_anomaly']
    }), 200


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """Выход из системы"""
    # Деактивация сессии
    active_session = UserSession.query.filter_by(
        user_id=session['user_id'],
        is_active=True
    ).first()

    if active_session:
        active_session.is_active = False
        db.session.commit()

    session.clear()
    return jsonify({'message': 'Вы вышли из системы'}), 200


@app.route('/api/profile-stats')
@login_required
def profile_stats():
    """Статистика профиля пользователя"""
    user_id = session['user_id']

    profiles = KeystrokeProfile.query.filter_by(user_id=user_id).all()
    events = KeystrokeEvent.query.filter_by(user_id=user_id).count()

    if profiles:
        profile = profiles[0]
        features = json.loads(profile.features)

        stats = {
            'profiles_count': len(profiles),
            'keystroke_events': events,
            'avg_typing_speed': features.get('typing_speed', 0),
            'avg_dwell_time': np.mean(features.get('dwell_times', [0])),
            'pressure_consistency': features.get('pressure_consistency', 0)
        }
    else:
        stats = {
            'profiles_count': 0,
            'keystroke_events': 0,
            'avg_typing_speed': 0,
            'avg_dwell_time': 0,
            'pressure_consistency': 0
        }

    return jsonify(stats), 200


# ========================
# ИНИЦИАЛИЗАЦИЯ
# ========================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)