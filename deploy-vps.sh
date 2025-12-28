#!/bin/bash
# Деплой ReviewScope на VPS (Ubuntu 22.04)

set -e

echo "🚀 ReviewScope VPS Deploy Script"
echo "================================="

# 1. Обновить систему
echo "📦 Updating system..."
sudo apt update && sudo apt upgrade -y

# 2. Установить Docker
echo "🐳 Installing Docker..."
sudo apt install -y docker.io docker-compose git
sudo systemctl enable docker
sudo systemctl start docker

# 3. Добавить пользователя в группу docker
sudo usermod -aG docker $USER

# 4. Клонировать репозиторий
echo "📥 Cloning repository..."
cd ~
if [ -d "reviewscope-production" ]; then
    cd reviewscope-production
    git pull
else
    git clone https://github.com/YOUR_USERNAME/reviewscope-production.git
    cd reviewscope-production
fi

# 5. Создать .env файл
echo "⚙️  Setting up environment..."
if [ ! -f "backend/.env" ]; then
    echo "Creating backend/.env - PLEASE EDIT IT!"
    cp backend/.env.example backend/.env
    echo "❗ Edit backend/.env and add your OPENAI_API_KEY"
    exit 1
fi

# 6. Собрать и запустить Docker
echo "🏗️  Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# 7. Проверить статус
echo "✅ Checking health..."
sleep 10
curl -f http://localhost:8888/health || echo "❌ Health check failed"

# 8. Показать логи
echo ""
echo "📋 Logs:"
docker-compose logs --tail=50

echo ""
echo "✅ Deployment complete!"
echo "📡 API running on http://YOUR_IP:8888"
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop: docker-compose down"
echo ""
echo "⚠️  Don't forget to:"
echo "   1. Set up firewall: ufw allow 8888"
echo "   2. Set up Nginx reverse proxy (optional)"
echo "   3. Set up SSL with certbot (optional)"
