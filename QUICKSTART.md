# ⚡ Быстрый старт для деплоя ReviewScope

## 🎯 Самый простой способ (Railway.app)

**Почему Railway:**
- ✅ $5/месяц за 8GB RAM (достаточно для моделей)
- ✅ Автодеплой из GitHub
- ✅ Легко настроить
- ✅ Поддерживает большие Docker образы

### Шаг 1: Подготовь репозиторий (5 минут)

```bash
cd c:/reviewscope-production

# Убедись, что .env НЕ в git (проверь .gitignore)
cat .gitignore | grep ".env"  # Должно быть: .env

# Создай репо
git init
git add .
git commit -m "Initial commit: ReviewScope production"

# Залей на GitHub
# Создай новый приватный репо на github.com
git remote add origin https://github.com/YOUR_USERNAME/reviewscope-production.git
git branch -M main
git push -u origin main
```

⚠️ **ВАЖНО**: Модели (~2.2GB) НЕ попадут в git (они в `.gitignore`). Это правильно!

---

### Шаг 2: Деплой на Railway (10 минут)

1. **Зайди на [railway.app](https://railway.app)**
   - Войди через GitHub

2. **Нажми "New Project"**
   - Deploy from GitHub repo
   - Выбери `reviewscope-production`

3. **Railway автоматически:**
   - Обнаружит Dockerfile
   - Начнёт сборку

4. **Добавь Environment Variables:**
   Нажми на сервис → Variables → Add Variable:
   ```
   OPENAI_API_KEY=sk-proj-q7-4SnqN5lf... (твой ключ)
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o-mini
   EMB_MODEL=text-embedding-3-small
   DEVICE=cpu
   SENT_MODEL=./models/sentiment/final
   RATE_MODEL=./models/rating/final
   PORT=8888
   ```

5. **⚠️ ПРОБЛЕМА: Модели не в Git!**

   Есть 2 решения:

   **А) Загрузи модели в Railway Volume (рекомендую):**
   ```bash
   # Установи Railway CLI
   npm i -g @railway/cli

   # Логинься
   railway login

   # Подключись к проекту
   railway link

   # Загрузи модели
   railway volume create models
   railway volume mount models /app/models

   # Теперь загрузи файлы (через SSH или CLI)
   ```

   **Б) Используй Hugging Face Hub:**
   - Загрузи модели на HF Hub
   - Используй `Dockerfile.light`
   - Установи env var: `HF_MODEL_REPO=your-username/reviewscope-models`

6. **Запусти сервис**
   - Railway даст тебе URL: `https://reviewscope-production.up.railway.app`

---

### Шаг 3: Проверь работу (2 минуты)

```bash
# Health check
curl https://your-app.up.railway.app/health

# Должен вернуть:
{
  "status": "healthy",
  "active_sessions": 0,
  "reviewscope_path": "/app/reviewscope_all.py",
  "reviewscope_exists": true,
  "config": {
    "sent_model": "./models/sentiment/final",
    "rate_model": "./models/rating/final",
    "llm_provider": "openai",
    "device": "cpu"
  }
}
```

---

## 🆓 Альтернатива: Render.com (бесплатно, но медленно)

### Быстрый вариант:

1. Зайди на [render.com](https://render.com)
2. New → Web Service → Connect GitHub
3. Выбери репозиторий
4. Settings:
   - **Runtime**: Docker
   - **Docker Command**: оставь пустым (берётся из Dockerfile)
   - **Plan**: Starter (free) или Starter Plus ($7)
5. Environment Variables (те же, что выше)
6. Create Web Service

⚠️ **Проблема**: Бесплатный tier засыпает через 15 минут без активности

---

## 📊 Сравнение хостингов

| Хостинг | Цена | RAM | Плюсы | Минусы |
|---------|------|-----|-------|--------|
| **Railway** | $5/мес | 8GB | ✅ Легко, быстро | ❌ Платно |
| **Render Free** | $0 | 512MB | ✅ Бесплатно | ❌ Медленно, засыпает |
| **Render Plus** | $7/мес | 2GB | ✅ Не засыпает | ❌ Дороже Railway |
| **Fly.io** | ~$3-5/мес | 1GB | ✅ Дешево | ❌ Сложная настройка |
| **VPS** | $4-6/мес | 2-4GB | ✅ Полный контроль | ❌ Нужен опыт |

---

## 🎯 Моя рекомендация:

**Для продакшена:** Railway.app ($5/мес)
- Просто работает
- Достаточно RAM для моделей
- Автодеплой

**Для тестов:** Локально
- `cd backend && python main.py`
- Бесплатно, быстро

---

## ❓ FAQ

### "Модели не найдены на Railway"
→ Загрузи модели через Railway Volume или используй Hugging Face Hub

### "Out of memory"
→ Увеличь RAM план до 8GB (Railway) или используй VPS

### "Сборка Docker занимает 20 минут"
→ Это нормально для образа с моделями ~3GB

### "Как обновить код?"
```bash
git add .
git commit -m "Update"
git push

# Railway/Render автоматически задеплоят новую версию
```

---

## 🚀 Готово!

После деплоя API будет доступно по адресу:
- Railway: `https://your-project.up.railway.app`
- Render: `https://your-service.onrender.com`

Теперь подключай фронтенд к этому URL! 🎉
