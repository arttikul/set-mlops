# Fedorenko | MLOps — ДЗ

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
5. Експортуємо результат: Export → JSON. Файл кладемо в `dataset/`.

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
