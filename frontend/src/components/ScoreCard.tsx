/**
 * Score Card Component - displays the main rating
 */

import clsx from 'clsx';

interface ScoreCardProps {
  score: number;
  maxScore?: number;
  note?: string;
  className?: string;
}

export function ScoreCard({ score, maxScore = 5, note, className }: ScoreCardProps) {
  const percentage = (score / maxScore) * 100;

  // Color based on score
  const getScoreColor = (s: number) => {
    if (s >= 4) return 'text-green-400';
    if (s >= 3) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStrokeColor = (s: number) => {
    if (s >= 4) return '#22c55e';
    if (s >= 3) return '#eab308';
    return '#ef4444';
  };

  return (
    <div className={clsx('card p-6 flex flex-col items-center', className)}>
      {/* Circular progress */}
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-dark-border"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={getStrokeColor(score)}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${percentage * 2.83} 283`}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        {/* Score text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={clsx('text-3xl font-bold', getScoreColor(score))}>
            {score.toFixed(1)}
          </span>
          <span className="text-sm text-dark-muted">из {maxScore}</span>
        </div>
      </div>

      {/* Label */}
      <div className="mt-4 text-center">
        <p className="text-lg font-medium">Реальный рейтинг</p>
        {note && <p className="text-sm text-dark-muted mt-1">{note}</p>}
      </div>
    </div>
  );
}

export default ScoreCard;
