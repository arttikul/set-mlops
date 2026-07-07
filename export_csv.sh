#!/bin/bash
# Експорт розміченого датасету з Label Studio у форматі CSV.
#
# На відміну від export_yolo.sh з лекції, тут не потрібно дотягувати файли
# з S3/локального сховища Label Studio — текст лежить прямо в анотації,
# тож CSV-експорту вже достатньо для готового датасету.
#
# Використання:
#   export LS_TOKEN=<ваш Personal Access Token>
#   ./export_csv.sh [project_id]
#
# Залежності: curl, python3.
set -e

# --- Налаштування (за потреби перевизначаються змінними середовища) ---
LS_URL="${LABEL_STUDIO_URL:-http://localhost:8080}"
OUTPUT_DIR="dataset"                 # тека з готовим датасетом (DVC-трекнута)
OUTPUT_FILE="$OUTPUT_DIR/emotion-dataset-labeled.csv"

PROJECT_ID="${1:-1}"                 # ID проєкту (за замовчуванням 1)

LS_TOKEN="${LS_TOKEN:-}"
if [ -z "$LS_TOKEN" ]; then
  echo "Помилка: не задано токен."
  echo "Створіть Personal Access Token у Label Studio (Account & Settings) і виконайте:"
  echo "  export LS_TOKEN=<ваш токен>"
  echo "  ./export_csv.sh [project_id]"
  exit 1
fi

# --- 1. Обмінюємо PAT на короткоживучий access-токен ---
echo "Отримуємо access-токен..."
ACCESS_TOKEN=$(curl -s -X POST "$LS_URL/api/token/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh\": \"$LS_TOKEN\"}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('access', ''))")
if [ -z "$ACCESS_TOKEN" ]; then
  echo "Помилка: не вдалося отримати access-токен."
  echo "Перевірте правильність токена та доступність Label Studio ($LS_URL)."
  exit 1
fi
AUTH="Authorization: Bearer $ACCESS_TOKEN"

# --- 2. Створюємо снапшот експорту ---
echo "Створюємо снапшот експорту для проєкту #$PROJECT_ID..."
EXPORT_ID=$(curl -s -X POST "$LS_URL/api/projects/$PROJECT_ID/exports/" \
  -H "$AUTH" -H "Content-Type: application/json" -d '{}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")
if [ -z "$EXPORT_ID" ]; then
  echo "Помилка: не вдалося створити снапшот експорту."
  exit 1
fi

# --- 3. Конвертуємо снапшот у формат CSV ---
echo "Конвертуємо снапшот #$EXPORT_ID у формат CSV..."
curl -s -X POST "$LS_URL/api/projects/$PROJECT_ID/exports/$EXPORT_ID/convert" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"export_type":"CSV"}' > /dev/null

# --- 4. Завантажуємо CSV (з кількома спробами, конвертація фонова) ---
echo "Завантажуємо CSV-експорт..."
rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"
for attempt in 1 2 3 4 5; do
  HTTP_CODE=$(curl -s -w "%{http_code}" -H "$AUTH" \
    "$LS_URL/api/projects/$PROJECT_ID/exports/$EXPORT_ID/download?exportType=CSV" \
    -o "$OUTPUT_FILE")
  if [ "$HTTP_CODE" = "200" ] && [ -s "$OUTPUT_FILE" ]; then
    break
  fi
  echo "  спроба $attempt: ще не готово (HTTP $HTTP_CODE), чекаємо..."
  sleep 2
done
if [ ! -s "$OUTPUT_FILE" ]; then
  echo "Помилка: не вдалося завантажити коректний CSV."; exit 1
fi

rows=$(($(wc -l < "$OUTPUT_FILE") - 1))
echo "Готово. Датасет у '$OUTPUT_FILE': $rows розмічених прикладів."
