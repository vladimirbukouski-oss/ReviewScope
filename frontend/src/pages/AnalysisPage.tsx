/**
 * Analysis Page - shows analysis results with tabs
 */

import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import clsx from 'clsx';
import { useAnalysis } from '../hooks/useAnalysis';
import { useAnalysisStore } from '../store/analysisStore';
import {
  ScoreCard,
  SentimentBar,
  ReviewCard,
  LoadingOverlay,
  ChatSection,
} from '../components';

const TABS = [
  { key: 'summary', label: '📊 Сводка', icon: '📊' },
  { key: 'reviews', label: '💬 Отзывы', icon: '💬' },
  { key: 'chat', label: '🤖 Чат', icon: '🤖' },
] as const;

export function AnalysisPage() {
  const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
  const { loadAnalysis, loadMoreReviews, product, summary, reviews, isLoading, error, status } = useAnalysis();
  const { activeTab, setActiveTab, toggleFavorite, history } = useAnalysisStore();

  const currentHistoryItem = history.find((h) => h.sessionId === urlSessionId);
  const isFavorite = currentHistoryItem?.isFavorite ?? false;

  // Load analysis on mount
  useEffect(() => {
    if (urlSessionId) {
      loadAnalysis(urlSessionId);
    }
  }, [urlSessionId, loadAnalysis]);

  // Show loading overlay
  if (isLoading && status && status.status !== 'ready') {
    return <LoadingOverlay status={status} />;
  }

  // Error state
  if (error && !product) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <div className="card p-8">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <span className="text-3xl">❌</span>
          </div>
          <h2 className="text-xl font-semibold mb-2">Ошибка загрузки</h2>
          <p className="text-dark-muted">{error}</p>
        </div>
      </div>
    );
  }

  // No data yet
  if (!product || !summary) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <div className="card p-8">
          <p className="text-dark-muted">Загрузка данных...</p>
        </div>
      </div>
    );
  }

  const sentimentMix = product.sentimentMix || { pos: 0.7, neu: 0.2, neg: 0.1 };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header with product info */}
      <div className="card p-6 mb-6">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                product.service === 'wb'
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'bg-orange-500/20 text-orange-400'
              }`}>
                {product.service === 'wb' ? 'Wildberries' : 'Onliner'}
              </span>
              <span className="text-dark-muted text-sm">
                {product.totalReviews} отзывов → {product.keptReviews} проанализировано
              </span>
            </div>
            <h1 className="text-xl md:text-2xl font-semibold mb-2">
              {summary.one_liner}
            </h1>
            <a
              href={product.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary-400 hover:underline truncate block"
            >
              {product.url}
            </a>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 ml-4">
            <button
              onClick={() => urlSessionId && toggleFavorite(urlSessionId)}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                isFavorite
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : 'bg-dark-card text-dark-muted hover:text-yellow-400'
              )}
              title={isFavorite ? 'Убрать из избранного' : 'Добавить в избранное'}
            >
              <svg className="w-5 h-5" fill={isFavorite ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
              </svg>
            </button>

            <button
              onClick={() => {
                const shareUrl = window.location.href;
                navigator.clipboard.writeText(shareUrl);
              }}
              className="p-2 rounded-lg bg-dark-card text-dark-muted hover:text-primary-400 transition-colors"
              title="Копировать ссылку"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={clsx(
              'px-6 py-3 rounded-xl font-medium whitespace-nowrap transition-all',
              activeTab === tab.key
                ? 'bg-primary-600 text-white'
                : 'bg-dark-card border border-dark-border hover:border-primary-500/50'
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'summary' && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Score and sentiment */}
            <div className="grid md:grid-cols-2 gap-6">
              <ScoreCard
                score={summary.score.value}
                maxScore={summary.score.scale}
                note={summary.score.note}
              />

              <div className="card p-6">
                <h3 className="font-semibold mb-4">Тональность отзывов</h3>
                <SentimentBar
                  positive={sentimentMix.pos}
                  neutral={sentimentMix.neu}
                  negative={sentimentMix.neg}
                />

                {product.suspiciousShare > 0.1 && (
                  <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                    <p className="text-sm text-yellow-400">
                      ⚠️ {(product.suspiciousShare * 100).toFixed(0)}% отзывов выглядят подозрительно
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Pros and Cons */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Pros */}
              <div className="card p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <span className="text-green-400">👍</span> Плюсы
                </h3>
                <ul className="space-y-3">
                  {summary.pros.map((pro, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-green-500/20 text-green-400 flex items-center justify-center flex-shrink-0 text-sm">
                        ✓
                      </span>
                      <div>
                        <p>{pro.point}</p>
                        {pro.evidence_ids.length > 0 && (
                          <p className="text-xs text-dark-muted mt-1">
                            {pro.evidence_ids.length} подтверждений
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Cons */}
              <div className="card p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <span className="text-red-400">👎</span> Минусы
                </h3>
                <ul className="space-y-3">
                  {summary.cons.map((con, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center flex-shrink-0 text-sm">
                        ✗
                      </span>
                      <div>
                        <p>{con.point}</p>
                        {con.evidence_ids.length > 0 && (
                          <p className="text-xs text-dark-muted mt-1">
                            {con.evidence_ids.length} подтверждений
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Aspects */}
            {summary.aspects.length > 0 && (
              <div className="card p-6">
                <h3 className="font-semibold mb-4">Детальный анализ</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  {summary.aspects.map((aspect, i) => (
                    <div key={i} className="p-4 bg-dark-bg rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{aspect.aspect}</h4>
                        <span className="text-xs text-dark-muted">
                          {aspect.mentions} упоминаний
                        </span>
                      </div>
                      <p className="text-sm text-dark-muted">{aspect.summary}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right column - Recommendations */}
          <div className="space-y-6">
            {/* Who it's for */}
            {summary.who_it_is_for.length > 0 && (
              <div className="card p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <span className="text-green-400">✅</span> Подойдёт
                </h3>
                <ul className="space-y-2">
                  {summary.who_it_is_for.map((who, i) => (
                    <li key={i} className="text-sm text-dark-muted">• {who}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Who should avoid */}
            {summary.who_should_avoid.length > 0 && (
              <div className="card p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <span className="text-red-400">⛔</span> Не подойдёт
                </h3>
                <ul className="space-y-2">
                  {summary.who_should_avoid.map((who, i) => (
                    <li key={i} className="text-sm text-dark-muted">• {who}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Confidence */}
            <div className="card p-6">
              <h3 className="font-semibold mb-2">Уверенность анализа</h3>
              <div className={clsx(
                'inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm',
                summary.confidence.level === 'high'
                  ? 'bg-green-500/20 text-green-400'
                  : summary.confidence.level === 'medium'
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : 'bg-red-500/20 text-red-400'
              )}>
                {summary.confidence.level === 'high' ? '🟢 Высокая' :
                 summary.confidence.level === 'medium' ? '🟡 Средняя' : '🔴 Низкая'}
              </div>
              <p className="text-sm text-dark-muted mt-2">{summary.confidence.why}</p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'reviews' && (
        <div className="space-y-4">
          {reviews.map((review) => (
            <ReviewCard key={review.id} review={review} />
          ))}

          {reviews.length > 0 && (
            <button
              onClick={() => loadMoreReviews()}
              className="w-full py-4 text-center text-primary-400 hover:text-primary-300 transition-colors"
            >
              Загрузить ещё
            </button>
          )}
        </div>
      )}

      {activeTab === 'chat' && urlSessionId && (
        <div className="card overflow-hidden">
          <ChatSection sessionId={urlSessionId} />
        </div>
      )}
    </div>
  );
}

export default AnalysisPage;
