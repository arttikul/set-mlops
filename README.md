# Fedorenko | MLOps — ДЗ 01

Розмітка та версіонування датасету для задачі **класифікації емоцій у тексті**
(6 класів: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`).

## Інструмент розмітки

**[Label Studio](https://labelstud.io/)**, тип проєкту — **Text Classification**
(один клас на приклад), з кастомним labeling-конфігом `.dataset/labeling_config.xml`:

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single">
    <Choice value="sadness"/>
    <Choice value="joy"/>
    <Choice value="love"/>
    <Choice value="anger"/>
    <Choice value="fear"/>
    <Choice value="surprise"/>
  </Choices>
</View>
```

Джерело даних — публічний датасет емоцій (`.dataset/emotion-dataset.csv`, пари попереньо форматований у json `data_example.json`).

На відміну від прикладу з зображеннями з лекції, тут не потрібне S3-сховище для
самих вихідних даних — текст завдання лежить прямо в JSON-файлі імпорту, файлових
посилань немає. SeaweedFS (S3-сумісне сховище) в `docker-compose.yml` піднімається
лише як бекенд для **версіонування датасету через DVC** (див. нижче).

## Як запустити розмітку

```bash
docker compose up -d
```

Це піднімає Label Studio (з Postgres як БД) і SeaweedFS (потрібен лише для DVC-стореджа).

- **Label Studio:** http://localhost:8080/

Кроки:
1. Реєструємо акаунт, створюємо проєкт.
2. Обираємо тип **Text Classification**, вставляємо labeling-конфіг з
   `labeling_config.xml` (Settings → Labeling Interface → Code).
3. Імпортуємо завдання: Import → завантажуємо попереднь форматовані файли
   (кожне завдання вже містить текст і попередню анотацію).
4. Переглядаємо/підтверджуємо розмітку в інтерфейсі проєкту.
5. Експортуємо результат у CSV скриптом `export_csv.sh` (аналог `export_yolo.sh`
   з лекції, без завантаження файлів — текст уже лежить в анотації):

```bash
export LS_TOKEN=<ваш Personal Access Token>   # Account & Settings → Personal Access Token
./export_csv.sh 1                             # ID проєкту, за замовчуванням 1
```

   Скрипт кладе результат у `dataset/emotion-dataset-labeled.csv`.

## Як працює версіонування датасету

Версіонується вміст теки `dataset/` (експорт із Label Studio) через **DVC**,
з віддаленим сховищем у SeaweedFS S3 (див. `.dvc/config`):

```
[core]
    remote = storage
['remote "storage"']
    url = s3://dvc-storage
    endpointurl = http://localhost:8333
```

Робочий цикл:

```bash
# після нового експорту з Label Studio в dataset/
dvc add dataset
git add dataset.dvc
git commit -m "Update dataset (vN)"
git tag vN
dvc push
```

Наявні версії:
- **v0** — порожній снепшот `dataset/` (до першого експорту, зафіксовано разом з
  ініціалізацією DVC).
- **v1** — перший реальний експорт з Label Studio: `project-1-...json`
  (5116 розмічених прикладів).
- **v2** — повний CSV-експорт через `export_csv.sh`: `emotion-dataset-labeled.csv`
  (105116 розмічених прикладів).
- **v3** — повторний CSV-експорт після продовження розмітки (155116 розмічених
  прикладів).

Відновлення конкретної версії:

```bash
git checkout v1
dvc checkout
```

Клонування репозиторію та отримання даних:

```bash
git clone https://github.com/arttikul/set-mlops
cd "Fedorenko | MLOps"
dvc pull   # тягне дані з s3://dvc-storage (http://localhost:8333)
```

Лінія даних: `emotion-dataset.csv` (сирі дані з мітками) →
`emotion_dataset_part_*.json` (завдання для імпорту) → Label Studio (розмітка/
перевірка) → `dataset/` (експорт) → DVC (`v0`, `v1`, …) → SeaweedFS S3 (`s3://dvc-storage`).

## Для яких задач плануються ці дані

Датасет буде використано для тренування моделі **класифікації емоцій у тексті**
(6 класів вище) у наступних домашніх роботах курсу — тренування/трекінг
експерименту, інференс (сервінг моделі) та моніторинг якості передбачень у
проді.

## Тренування моделі та трекінг експериментів (ДЗ-2)

**Модель:** TF-IDF (`scikit-learn`) + простий PyTorch MLP-класифікатор
(`Linear → ReLU → Dropout → Linear`) на 6 класів емоцій, натренований на
`dataset/emotion-dataset-labeled.csv` (v2, 105116 прикладів; 90/10
стратифікований train/val split). Це найпростіший локальний варіант з лекції
(за зразком `3-Model-Training-Tracking/lr/train.py`) — окремий Ray/K8s-кластер
не піднімався, він не потрібен для цього обсягу даних.

**Трекінг:** [Weights & Biases](https://wandb.ai/) — щоб зберегти наступність із
ДЗ-3 (`4-Inference/FastAPI_Docker/download_model.py` тягне модель саме з W&B
Model Registry).

### Як запустити тренування

```bash
cd "Fedorenko | MLOps/training"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # впишіть свій WANDB_API_KEY (https://wandb.ai/authorize) і WANDB_ENTITY

python train.py --run-name baseline
# інший порівнюваний запуск (інші гіперпараметри):
python train.py --run-name wide-hidden --hidden-dim 256 --lr 5e-4 --max-features 30000
```

Гіперпараметри (`--epochs`, `--lr`, `--hidden-dim`, `--max-features`, `--batch-size`,
`--dropout`) задаються прапорцями — кожен запуск логується окремим ран у W&B, тож
запуски з різними значеннями одразу можна порівняти.

`train.py` логує в W&B (`wandb.init`, `wandb.log` по кожній епосі, `wandb.watch`)
`train/val loss` і `accuracy`, а наприкінці — таблицю per-class precision/recall/F1
на валідації. Модель (`model.pt`), TF-IDF-векторайзер (`vectorizer.pkl`) і мапу
класів (`labels.json`) зберігає як `wandb.Artifact` і одразу лінкує нову версію в
**W&B Registry** (`wandb-registry-model/emotion-classifier` — новий формат
реєстру моделей, що прийшов на зміну старому `<entity>/model-registry/<name>`).

### Де дивитися результати

- **Проєкт W&B:** https://wandb.ai/arttikul-set-university/emotion-classification
  — на момент написання тут 4 запуски (`baseline`, `wide-hidden`, `baseline-rerun`
  на v2 датасеті з val accuracy ≈ 0.97, та `v3-full-dataset` на оновленому v3
  датасеті з val accuracy ≈ 0.95).
- **Реєстр моделей:** W&B → Registry → Model → колекція `emotion-classifier`
  (entity `arttikul-set-university`) — версії артефакта, готові до завантаження
  через `run.use_artifact('wandb-registry-model/emotion-classifier:latest')`.

## Інференс (ДЗ-3)

**Сервінг:** FastAPI + Docker (за зразком `4-Inference/FastAPI_Docker` з лекції).
Модель **не захардкоджена в репозиторії** — контейнер при старті сам тягне
останню версію з W&B Registry (`inference/download_model.py`,
`wandb-registry-model/emotion-classifier:latest`: `model.pt`, `vectorizer.pkl`,
`labels.json`) і лише потім піднімає сервер.

### Як підняти сервіс

```bash
cd "Fedorenko | MLOps/inference"
cp .env.example .env   # впишіть свій WANDB_API_KEY

docker build -t emotion-inference .
docker run --rm --env-file .env -p 8081:8080 emotion-inference
```

(Порт хоста `8081` — щоб не конфліктувати з Label Studio, який займає `8080`.)

### Як перевірити роботу

```bash
curl http://localhost:8081/ping
# {"status":"ok"}

curl -X POST http://localhost:8081/invocations \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["i am so happy today, everything feels wonderful", "i am terrified of what might happen next"]}'
```

Відповідь — список передбачень (мітка + ймовірності по кожному з 6 класів) для
кожного тексту з запиту, наприклад:

```json
[
  {"text": "i am so happy today, everything feels wonderful", "label": "joy", "probabilities": {"joy": 1.0, "...": 0.0}},
  {"text": "i am terrified of what might happen next", "label": "fear", "probabilities": {"fear": 0.9997, "...": 0.0}}
]
```

Без Docker (локально, для розробки): `pip install -r requirements.txt`,
`python download_model.py`, потім `uvicorn main:app --host 0.0.0.0 --port 8081`.
