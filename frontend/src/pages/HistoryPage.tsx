/**
 * History Page - shows all analyzed products
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAnalysisStore } from '../store/analysisStore';
import clsx from 'clsx';

type FilterType = 'all' | 'favorites' | 'wb' | 'onliner';

export function HistoryPage() {
  const { history, toggleFavorite, removeFromHistory, clearHistory } = useAnalysisStore();
  const [filter, setFilter] = useState<FilterType>('all');

  const filteredHistory = history.filter((item) => {
    switch (filter) {
      case 'favorites':
        return item.isFavorite;
      case 'wb':
        return item.service === 'wb';
      case 'onliner':
        return item.service === 'onliner';
      default:
        return true;
    }
  });

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">История анализов</h1>

        {history.length > 0 && (
          <button
            onClick={() => {
              if (confirm('Очистить всю историю?')) {
                clearHistory();
              }
            }}
            className="text-sm text-dark-muted hover:text-red-400 transition-colors"
          >
            Очистить историю
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {[
          { key: 'all', label: 'Все', count: history.length },
          { key: 'favorites', label: '★ Избранное', count: history.filter((h) => h.isFavorite).length },
          { key: 'wb', label: 'WB', count: history.filter((h) => h.service === 'wb').length },
          { key: 'onliner', label: 'Onliner', count: history.filter((h) => h.service === 'onliner').length },
        ].map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key as FilterType)}
            className={clsx(
              'px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors',
              filter === f.key
                ? 'bg-primary-600 text-white'
                : 'bg-dark-card border border-dark-border hover:border-primary-500/50'
            )}
          >
            {f.label} ({f.count})
          </button>
        ))}
      </div>

      {/* History list */}
      {filteredHistory.length === 0 ? (
        <div className="card p-12 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-dark-border flex items-center justify-center">
            <span className="text-3xl">📋</span>
          </div>
          <h3 className="text-lg font-medium mb-2">
            {filter === 'all' ? 'История пуста' : 'Ничего не найдено'}
          </h3>
          <p className="text-dark-muted">
            {filter === 'all'
              ? 'Проанализируйте товар, и он появится здесь'
              : 'Попробуйте изменить фильтр'}
          </p>
          {filter === 'all' && (
            <Link to="/" className="btn-primary inline-block mt-4">
              Анализировать товар
            </Link>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item) => (
            <div
              key={item.sessionId}
              className="card p-4 hover:border-primary-500/50 transition-colors group"
            >
              <div className="flex items-center justify-between">
                <Link
                  to={`/analysis/${item.sessionId}`}
                  className="flex-1 min-w-0"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`px-2 py-0.5 text-xs rounded ${
                        item.service === 'wb'
                          ? 'bg-purple-500/20 text-purple-400'
                          : 'bg-orange-500/20 text-orange-400'
                      }`}
                    >
                      {item.service === 'wb' ? 'WB' : 'Onliner'}
                    </span>
                    {item.isFavorite && <span className="text-yellow-400">★</span>}
                  </div>
                  <p className="font-medium truncate group-hover:text-primary-400 transition-colors">
                    {item.productName || item.url}
                  </p>
                  <p className="text-sm text-dark-muted">
                    {item.totalReviews} отзывов •{' '}
                    {new Date(item.createdAt).toLocaleDateString('ru-RU', {
                      day: 'numeric',
                      month: 'short',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </p>
                </Link>

                <div className="flex items-center gap-4 ml-4">
                  {/* Score */}
                  <div className="text-right">
                    <div className="text-2xl font-bold text-primary-400">
                      {item.truthStars?.toFixed(1) || '—'}
                    </div>
                    <div className="text-xs text-dark-muted">из 5</div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        toggleFavorite(item.sessionId);
                      }}
                      className={clsx(
                        'p-2 rounded-lg transition-colors',
                        item.isFavorite
                          ? 'text-yellow-400 hover:bg-yellow-500/20'
                          : 'text-dark-muted hover:text-yellow-400'
                      )}
                      title={item.isFavorite ? 'Убрать из избранного' : 'В избранное'}
                    >
                      <svg
                        className="w-5 h-5"
                        fill={item.isFavorite ? 'currentColor' : 'none'}
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                        />
                      </svg>
                    </button>

                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        if (confirm('Удалить из истории?')) {
                          removeFromHistory(item.sessionId);
                        }
                      }}
                      className="p-2 rounded-lg text-dark-muted hover:text-red-400 transition-colors"
                      title="Удалить"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
