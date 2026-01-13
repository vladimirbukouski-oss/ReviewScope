/**
 * Loading Overlay Component - shows during analysis with flying reviews
 */

import { useEffect, useState, useRef } from 'react';
import type { AnalysisStatus } from '../api/client';

interface LoadingOverlayProps {
  status: AnalysisStatus;
}

interface StreamReview {
  id: string;
  text: string;
  rating: number;
  user: string;
}

interface FlyingReview extends StreamReview {
  key: string;
  x: number;
  y: number;
  duration: number;
  delay: number;
  direction: 'left' | 'right';
}

const STEPS = [
  { key: 'fetching', label: 'Сбор', icon: '📥' },
  { key: 'scoring', label: 'Анализ', icon: '🔍' },
  { key: 'building_rag', label: 'Индекс', icon: '📊' },
  { key: 'summarizing', label: 'Сводка', icon: '✨' },
];

const API_BASE = import.meta.env.VITE_API_URL || '/api';

export function LoadingOverlay({ status }: LoadingOverlayProps) {
  const currentStepIndex = STEPS.findIndex((s) => s.key === status.status);
  const [flyingReviews, setFlyingReviews] = useState<FlyingReview[]>([]);
  const lastSeenRef = useRef(0);
  const reviewKeyRef = useRef(0);

  // Fetch reviews for animation
  useEffect(() => {
    if (!status.session_id || status.status === 'pending') return;

    const fetchReviews = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/reviews-stream/${status.session_id}?last_seen=${lastSeenRef.current}`
        );
        if (!res.ok) return;

        const data = await res.json();

        if (data.reviews && data.reviews.length > 0) {
          const newFlying: FlyingReview[] = data.reviews.map((r: StreamReview, i: number) => ({
            ...r,
            key: `review-${reviewKeyRef.current++}`,
            x: Math.random() * 80 + 10, // 10-90% from left
            y: Math.random() * 60 + 20, // 20-80% from top
            duration: 8 + Math.random() * 6, // 8-14 seconds
            delay: i * 0.3, // stagger
            direction: Math.random() > 0.5 ? 'left' : 'right',
          }));

          setFlyingReviews((prev) => [...prev, ...newFlying].slice(-20)); // Keep last 20
          lastSeenRef.current = data.next_index;
        }
      } catch {
        // Silently ignore errors
      }
    };

    fetchReviews();
    const interval = setInterval(fetchReviews, 2000);

    return () => clearInterval(interval);
  }, [status.session_id, status.status]);

  // Clean up old reviews
  useEffect(() => {
    const cleanup = setInterval(() => {
      setFlyingReviews((prev) => prev.slice(-15));
    }, 5000);

    return () => clearInterval(cleanup);
  }, []);

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-dark-bg/95 backdrop-blur-sm">
      {/* Flying reviews background */}
      <div className="absolute inset-0 pointer-events-none">
        {flyingReviews.map((review) => (
          <div
            key={review.key}
            className="flying-review absolute max-w-xs p-3 rounded-lg bg-dark-card/60 border border-dark-border/30 shadow-lg"
            style={{
              left: `${review.x}%`,
              top: `${review.y}%`,
              animation: `fly-${review.direction} ${review.duration}s linear ${review.delay}s forwards`,
              opacity: 0,
            }}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-dark-muted">{review.user}</span>
              <div className="flex">
                {[1, 2, 3, 4, 5].map((star) => (
                  <span
                    key={star}
                    className={`text-xs ${star <= review.rating ? 'text-yellow-400' : 'text-dark-border'}`}
                  >
                    ★
                  </span>
                ))}
              </div>
            </div>
            <p className="text-xs text-dark-text/80 line-clamp-3">{review.text}</p>
          </div>
        ))}
      </div>

      {/* Center content */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="max-w-md w-full mx-4 card p-8 text-center bg-dark-card/95 backdrop-blur-md shadow-2xl">
          {/* Animated icon */}
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center animate-pulse-glow">
            <svg
              className="w-10 h-10 text-white animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>

          {/* Progress */}
          <div className="mb-6">
            <div className="h-2 rounded-full bg-dark-border overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
                style={{ width: `${status.progress}%` }}
              />
            </div>
            <p className="mt-2 text-2xl font-bold gradient-text">{status.progress}%</p>
          </div>

          {/* Steps */}
          <div className="flex justify-center gap-6 mb-6">
            {STEPS.map((step, index) => (
              <div
                key={step.key}
                className={`flex flex-col items-center transition-all ${
                  index <= currentStepIndex ? 'opacity-100' : 'opacity-40'
                }`}
              >
                <span className="text-2xl mb-1">{step.icon}</span>
                <span
                  className={`text-xs ${
                    step.key === status.status
                      ? 'text-primary-400 font-medium'
                      : 'text-dark-muted'
                  }`}
                >
                  {step.label}
                </span>
              </div>
            ))}
          </div>

          {/* Message */}
          <p className="text-dark-text">{status.message}</p>

          {/* Review count */}
          {flyingReviews.length > 0 && (
            <p className="text-sm text-dark-muted mt-2">
              Обработано отзывов: {lastSeenRef.current}
            </p>
          )}

          {/* ETA */}
          {status.eta_seconds && status.eta_seconds > 0 && (
            <p className="text-sm text-dark-muted mt-2">
              Примерно {status.eta_seconds} сек.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default LoadingOverlay;
