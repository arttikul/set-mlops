## Локальний запуск Label Studio

```bash
docker compose up -d
```

### Налаштування SeaweedFS

SeaweedFS — це S3-сумісне об'єктне сховище. Воно надає кілька веб-інтерфейсів:

- **Admin UI** (керування бакетами, S3-користувачами, ключами): http://localhost:23646/
- **Filer UI** (файловий браузер по бакетах): http://localhost:8888/
- **Master UI** (стан кластера, томи): http://localhost:9333/

**S3 API:** http://localhost:8333
- **Access Key:** s3admin
- **Secret Key:** s3secret-changeme

> Облікові дані задано у файлі `docker/config/s3.json`.

Створіть три бакети:
- `cars-dataset` — для зберігання вихідних даних (зображення)
- `cars-labeled-dataset` — для зберігання розмічених даних від Label Studio
- `cars-dvc-storage` — для зберігання версіонованих даних через DVC

Найпростіше створити бакети через AWS CLI (SeaweedFS повністю S3-сумісний).

Спочатку один раз додаємо окремий AWS-профіль `mlops-set-local` для локального сховища
(потребує AWS CLI v2.13+, бо `endpoint_url` задається прямо в профілі):

```bash
aws configure set profile.mlops-set-local.region us-east-1
aws configure set profile.mlops-set-local.endpoint_url http://localhost:8333
aws configure set aws_access_key_id s3admin --profile mlops-set-local
aws configure set aws_secret_access_key s3secret-changeme --profile mlops-set-local
```

Далі працюємо просто через `--profile mlops-set-local` (endpoint береться з профілю):

```bash
# Створюємо бакети
aws --profile mlops-set-local s3 mb s3://cars-dataset
aws --profile mlops-set-local s3 mb s3://cars-labeled-dataset
aws --profile mlops-set-local s3 mb s3://cars-dvc-storage

# Перевіряємо список бакетів
aws --profile mlops-set-local s3 ls
```

Створені бакети також видно у Filer UI (http://localhost:8888/buckets/) та в Admin UI.

### Налаштування Label Studio

**Label Studio:** http://localhost:8080/

1. Реєструємо акаунт
2. Створюємо проєкт
   - Обираємо тип: Object Detection with Bounding Boxes

3. Налаштовуємо проєкт
   - Обираємо Cloud Storage
   - Налаштовуємо Source Cloud Storage (бакет `cars-dataset`) та Target Cloud
     Storage (бакет `cars-labeled-dataset`):
     - Storage Type: AWS S3
     - S3 Endpoint: http://seaweedfs:8333 (звертаємось до SeaweedFS за іменем
       сервісу в Docker-мережі, а не localhost)
     - Access Key ID: s3admin
     - Secret Access Key: s3secret-changeme
     - Опція "Treat every bucket object as a source file" - щоб кожен файл був завданням
     - Вимикаємо pre-signed URLs - для проксювання зображень на фронт

4. Завантажуємо зображення
5. Виконуємо демо розмітки
6. Показуємо приклад для розмітки питання-відповідь

#### Локальне завантаження vs робота через S3

Важливо розуміти різницю між двома способами додавання зображень — від цього
залежить, який шлях до картинки опиниться в анотації.

**Якщо завантажуємо картинку напряму в Label Studio** (Import / drag-and-drop):
- файл зберігається **локально всередині Label Studio**, а не в S3;
- LS додає випадковий префікс до імені (наприклад `9.png` → `056cc710-9.png`);
- у завданні шлях локальний, і в анотації буде:
  ```json
  "data": { "image": "/data/upload/1/056cc710-9.png" },
  "file_upload": 1
  ```
- картинка доступна за адресою `http://localhost:8080/data/upload/1/056cc710-9.png`
  (а фізично — у томі Label Studio: `/label-studio/data/media/upload/...`);
- у Target Storage експортується **тільки JSON-анотація**, саме зображення в S3
  не потрапляє.

**Якщо працюємо через Source Cloud Storage** (бакет `cars-dataset`):
- зображення лишаються в S3, завдання лише посилається на оригінал;
- імʼя файлу не змінюється, і в анотації одразу правильний S3-шлях:
  ```json
  "data": { "image": "s3://cars-dataset/6.png" },
  "file_upload": null
  ```
- лінія даних «оригінал → анотація» цілісна — це і є правильний підхід для MLOps.