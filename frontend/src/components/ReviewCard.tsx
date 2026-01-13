/**
 * Review Card Component
 */

import clsx from 'clsx';
import type { Review } from '../api/client';

interface ReviewCardProps {
  review: Review;
  isEvidence?: boolean;
  className?: string;
}

export function ReviewCard({ review, isEvidence, className }: ReviewCardProps) {
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'pos':
        return 'bg-green-500/20 text-green-400';
      case 'neg':
        return 'bg-red-500/20 text-red-400';
      default:
        return 'bg-yellow-500/20 text-yellow-400';
    }
  };

  const getTrustColor = (trust: number) => {
    if (trust >= 0.7) return 'text-green-400';
    if (trust >= 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className={clsx('card p-4 animate-slide-up', className)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          {/* Avatar */}
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center text-white font-medium">
            {review.user.charAt(0).toUpperCase()}
          </div>
          <div>
            <p className="font-medium">{review.user}</p>
            <p className="text-sm text-dark-muted">{review.created}</p>
          </div>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-2">
          {isEvidence && (
            <span className="px-2 py-0.5 text-xs bg-primary-500/20 text-primary-400 rounded">
              📎 Доказательство
            </span>
          )}
          <span className={clsx('px-2 py-0.5 text-xs rounded', getSentimentColor(review.sentiment))}>
            {review.sentiment === 'pos' ? '👍' : review.sentiment === 'neg' ? '👎' : '😐'}
          </span>
        </div>
      </div>

      {/* Rating and Trust */}
      <div className="flex items-center gap-4 mb-3">
        {/* Stars */}
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <svg
              key={star}
              className={clsx(
                'w-4 h-4',
                star <= review.rating ? 'text-yellow-400' : 'text-dark-border'
              )}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          ))}
        </div>

        {/* Trust score */}
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-dark-muted">Доверие:</span>
          <span className={clsx('text-sm font-medium', getTrustColor(review.trust))}>
            {(review.trust * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Text */}
      <p className="text-dark-text leading-relaxed">{review.text}</p>
    </div>
  );
}

export default ReviewCard;
