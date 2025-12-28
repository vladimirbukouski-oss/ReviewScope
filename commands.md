# 📝 Шпаргалка команд ReviewScope

## 🐳 Docker (локальная разработка)

```bash
# Собрать и запустить
docker-compose up --build

# Запустить в фоне
docker-compose up -d

# Остановить
docker-compose down

# Посмотреть логи
docker-compose logs -f backend

# Пересобрать после изменений
docker-compose build --no-cache
docker-compose up
```

## 🚂 Railway.app

```bash
# Установить CLI
npm i -g @railway/cli

# Войти
railway login

# Подключиться к проекту
railway link

# Посмотреть логи
railway logs

# Открыть проект в браузере
railway open

# Загрузить модели в volume
railway volume create models
railway volume mount models /app/models
# Затем загрузи файлы через dashboard
```

## 🎨 Render.com

```bash
# Через веб-интерфейс:
# 1. Dashboard → твой сервис
# 2. Logs — посмотреть логи
# 3. Environment — изменить переменные
# 4. Manual Deploy — пересобрать вручную
```

## 🧪 Локальный запуск (без Docker)

```bash
# Установить зависимости
cd backend
pip install -r requirements.txt

# Запустить backend
python main.py

# Backend будет на http://localhost:8888
```

## 🔍 Тестовые запросы

```bash
# Health check
curl http://localhost:8888/health

# Анализ товара
curl -X POST http://localhost:8888/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.wildberries.ru/catalog/181425009",
    "use_cache": true
  }'

# Получить статус
curl http://localhost:8888/status/{session_id}

# Получить результаты
curl http://localhost:8888/summary/{session_id}

# Задать вопрос через RAG
curl -X POST http://localhost:8888/chat/{session_id} \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "question": "Какие основные минусы товара?"
  }'
```

## 🔧 Git

```bash
# Статус
git status

# Добавить изменения
git add .

# Commit
git commit -m "Update: улучшен промпт для RAG"

# Push (автодеплой на Railway/Render)
git push origin main

# Создать новую ветку
git checkout -b feature/new-prompt

# Слить ветку
git checkout main
git merge feature/new-prompt
```

## 📦 Standalone скрипт (без backend)

```bash
# Полный анализ
python reviewscope_all.py run \
  --url "https://www.wildberries.ru/catalog/123456789" \
  --out_dir stage3_out \
  --sent_model models/sentiment/final \
  --rate_model models/rating/final \
  --device cpu \
  --fb_from 1 --fb_to 2 \
  --make_summary

# Только сбор и скоринг (Stage3)
python reviewscope_all.py stage3 \
  --url "https://www.wildberries.ru/catalog/123456789" \
  --out_dir stage3_out \
  --sent_model models/sentiment/final \
  --rate_model models/rating/final

# Построить RAG индекс
python reviewscope_all.py rag_build \
  --bundle stage3_out/stage3_bundle.json \
  --rag_dir stage3_out/rag

# Задать вопрос через RAG
python reviewscope_all.py ask \
  --rag_dir stage3_out/rag \
  --question "Какие проблемы с размером?"

# Сгенерировать summary
python reviewscope_all.py summarize \
  --bundle stage3_out/stage3_bundle.json \
  --out stage4_summary.json
```

## 🛠️ Отладка

```bash
# Проверить модели
ls -lh models/sentiment/final/
ls -lh models/rating/final/

# Проверить .env
cat backend/.env

# Проверить логи Docker
docker logs reviewscope-backend-1

# Проверить использование памяти
docker stats

# Зайти внутрь контейнера
docker exec -it reviewscope-backend-1 /bin/bash

# Проверить Python зависимости
pip list | grep -E "torch|transformers|fastapi"
```

## 🧹 Очистка

```bash
# Удалить старые Docker образы
docker system prune -a

# Удалить кэш Python
find . -type d -name "__pycache__" -exec rm -rf {} +

# Удалить старые данные
rm -rf backend/data/*
```
