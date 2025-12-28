# 🚀 Деплой ReviewScope на хостинг

## Проблема: Модели занимают 2.2GB

У нас есть два подхода:

### **Вариант 1: Полный Docker образ (с моделями)**
- ✅ Быстрый старт
- ❌ Большой размер образа (~3GB)
- ❌ Долгая загрузка на Render/Railway

### **Вариант 2: Модели из облака**
- ✅ Легкий образ (~500MB)
- ✅ Быстрая загрузка
- ❌ Нужно хранить модели в S3/R2/HF Hub

---

## 📦 Вариант 1: Render.com (с моделями в образе)

### Шаг 1: Создай Git репозиторий
```bash
cd c:/reviewscope-production
git init
git add .
git commit -m "Initial commit"

# Создай репо на GitHub и залей:
git remote add origin https://github.com/YOUR_USERNAME/reviewscope-production.git
git branch -M main
git push -u origin main
```

### Шаг 2: Деплой на Render
1. Зайди на [render.com](https://render.com)
2. Нажми "New +" → "Web Service"
3. Подключи GitHub репозиторий
4. Выбери:
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Starter (free) или Starter Plus ($7/мес для 2GB RAM)

### Шаг 3: Настрой Environment Variables
```
OPENAI_API_KEY=sk-your-actual-key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
DEVICE=cpu
SENT_MODEL=./models/sentiment/final
RATE_MODEL=./models/rating/final
```

### Шаг 4: Deploy!
Нажми "Create Web Service" — Render автоматически:
- Соберёт Docker образ
- Запустит на порту 8888
- Даст публичный URL: `https://reviewscope-api.onrender.com`

⚠️ **Важно**:
- Сборка займёт ~10-15 минут (из-за размера моделей)
- Бесплатный tier может быть медленным, рекомендую Starter Plus ($7/мес)

---

## ☁️ Вариант 2: Модели из Cloudflare R2 (дешевле)

### Шаг 1: Загрузи модели в R2/S3
```bash
# Установи rclone или aws cli
# Загрузи модели:
aws s3 sync models/ s3://your-bucket/reviewscope/models/

# Или используй Cloudflare R2 (бесплатно до 10GB)
```

### Шаг 2: Создай скрипт загрузки
Используй `Dockerfile.light` и обнови `download_models.py`:

```python
import boto3
import os

s3 = boto3.client('s3',
    endpoint_url=os.getenv('R2_ENDPOINT'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('R2_SECRET_KEY')
)

# Download models from R2
s3.download_file('your-bucket', 'reviewscope/models/sentiment/final/model.safetensors',
                 './models/sentiment/final/model.safetensors')
# ... repeat for all model files
```

### Шаг 3: Деплой на Render
- Используй `Dockerfile.light`
- Добавь env vars: `R2_ENDPOINT`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`

---

## 🐋 Вариант 3: VPS (DigitalOcean/Hetzner)

### Подходит если:
- Нужен полный контроль
- Бюджет ~$4-6/месяц
- Не боишься SSH

### Быстрый деплой:
```bash
# На VPS (Ubuntu 22.04):
sudo apt update && sudo apt install -y docker.io docker-compose

# Клонируй репо
git clone https://github.com/YOUR_USERNAME/reviewscope-production.git
cd reviewscope-production

# Создай .env
cp backend/.env.example backend/.env
nano backend/.env  # заполни API ключи

# Запусти
docker build -t reviewscope .
docker run -d -p 8888:8888 --env-file backend/.env reviewscope

# Настрой Nginx reverse proxy (опционально)
```

---

## 🎯 Рекомендации

### Для продакшена:
1. **Railway.app** ($5-7/мес) — лучший баланс цена/качество
2. **Fly.io** — если нужна production-grade инфраструктура
3. **VPS** — если хочешь полный контроль

### Для тестов:
1. **Render.com Free** — но будет медленно
2. **Локально** — самый быстрый вариант

---

## 🔧 После деплоя

Проверь работоспособность:
```bash
# Health check
curl https://your-app.onrender.com/health

# Тестовый запрос
curl -X POST https://your-app.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.wildberries.ru/catalog/181425009"}'
```

---

## ❓ Частые проблемы

### "Out of memory" на Render Free
→ Используй Starter Plus ($7/мес) с 2GB RAM

### "Build timeout"
→ Модели слишком большие, используй Вариант 2 (облако)

### "Models not found"
→ Проверь пути в `.env`: `SENT_MODEL=./models/sentiment/final`

### "OpenAI API error"
→ Проверь `OPENAI_API_KEY` в Environment Variables
