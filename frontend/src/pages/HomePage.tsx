/**
 * Home Page - URL input and recent analyses
 */

import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useAnalysis } from '../hooks/useAnalysis';
import { useAnalysisStore } from '../store/analysisStore';

export function HomePage() {
  const [url, setUrl] = useState('');
  const { startAnalysis, isLoading, error } = useAnalysis();
  const { history } = useAnalysisStore();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      startAnalysis(url.trim());
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      {/* Hero section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="gradient-text">AI-анализ отзывов</span>
        </h1>
        <p className="text-lg text-dark-muted max-w-2xl mx-auto">
          Вставьте ссылку на товар с Wildberries или Onliner и получите реальный рейтинг,
          плюсы, минусы и AI-сводку на основе анализа всех отзывов
        </p>
      </div>

      {/* URL input */}
      <form onSubmit={handleSubmit} className="mb-12">
        <div className="card p-2 flex gap-2">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Вставьте ссылку на товар..."
            className="flex-1 px-4 py-3 bg-transparent focus:outline-none text-lg"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!url.trim() || isLoading}
            className="btn-primary px-8 py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Анализ...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Анализировать
              </>
            )}
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div className="mt-4 p-4 bg-red-500/20 text-red-400 rounded-xl">
            {error}
          </div>
        )}

        {/* Supported platforms */}
        <p className="text-center text-sm text-dark-muted mt-4">
          Поддерживаются: Wildberries, Onliner
        </p>
      </form>

      {/* Recent analyses from history */}
      {history.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            История анализов
          </h2>

          <div className="grid gap-4">
            {history.slice(0, 5).map((item) => (
              <Link
                key={item.sessionId}
                to={`/analysis/${item.sessionId}`}
                className="card p-4 hover:border-primary-500/50 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-0.5 text-xs rounded ${
                        item.service === 'wb'
                          ? 'bg-purple-500/20 text-purple-400'
                          : 'bg-orange-500/20 text-orange-400'
                      }`}>
                        {item.service === 'wb' ? 'WB' : 'Onliner'}
                      </span>
                      {item.isFavorite && (
                        <span className="text-yellow-400">★</span>
                      )}
                    </div>
                    <p className="font-medium truncate group-hover:text-primary-400 transition-colors">
                      {item.productName || item.url}
                    </p>
                    <p className="text-sm text-dark-muted">
                      {item.totalReviews} отзывов • {new Date(item.createdAt).toLocaleDateString('ru-RU')}
                    </p>
                  </div>

                  <div className="text-right ml-4">
                    <div className="text-2xl font-bold text-primary-400">
                      {item.truthStars?.toFixed(1) || '—'}
                    </div>
                    <div className="text-xs text-dark-muted">из 5</div>
                  </div>
                </div>
              </Link>
            ))}
          </div>

          {history.length > 5 && (
            <Link
              to="/history"
              className="block text-center text-primary-400 hover:text-primary-300 mt-4 text-sm"
            >
              Показать все ({history.length})
            </Link>
          )}
        </div>
      )}

      {/* Features */}
      <div className="mt-16 grid md:grid-cols-3 gap-6">
        <div className="card p-6 text-center">
          <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-green-500/20 flex items-center justify-center">
            <span className="text-2xl">🎯</span>
          </div>
          <h3 className="font-semibold mb-2">Реальный рейтинг</h3>
          <p className="text-sm text-dark-muted">
            Оценка на основе AI-анализа, а не накрученных отзывов
          </p>
        </div>

        <div className="card p-6 text-center">
          <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-primary-500/20 flex items-center justify-center">
            <span className="text-2xl">📊</span>
          </div>
          <h3 className="font-semibold mb-2">Плюсы и минусы</h3>
          <p className="text-sm text-dark-muted">
            Автоматическое выделение достоинств и недостатков
          </p>
        </div>

        <div className="card p-6 text-center">
          <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-purple-500/20 flex items-center justify-center">
            <span className="text-2xl">💬</span>
          </div>
          <h3 className="font-semibold mb-2">AI-чат</h3>
          <p className="text-sm text-dark-muted">
            Задавайте вопросы и получайте ответы с доказательствами
          </p>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
