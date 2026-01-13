/**
 * Sentiment Bar Component - shows positive/neutral/negative breakdown
 */

interface SentimentBarProps {
  positive: number;
  neutral: number;
  negative: number;
  className?: string;
}

export function SentimentBar({ positive, neutral, negative, className }: SentimentBarProps) {
  const total = positive + neutral + negative;
  const posPercent = total > 0 ? (positive / total) * 100 : 33;
  const neuPercent = total > 0 ? (neutral / total) * 100 : 34;
  const negPercent = total > 0 ? (negative / total) * 100 : 33;

  return (
    <div className={className}>
      {/* Bar */}
      <div className="h-3 rounded-full overflow-hidden flex bg-dark-border">
        <div
          className="bg-green-500 transition-all duration-500"
          style={{ width: `${posPercent}%` }}
        />
        <div
          className="bg-yellow-500 transition-all duration-500"
          style={{ width: `${neuPercent}%` }}
        />
        <div
          className="bg-red-500 transition-all duration-500"
          style={{ width: `${negPercent}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between mt-2 text-sm">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
          <span className="text-dark-muted">
            Позитивные <span className="text-green-400 font-medium">{posPercent.toFixed(0)}%</span>
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
          <span className="text-dark-muted">
            Нейтральные <span className="text-yellow-400 font-medium">{neuPercent.toFixed(0)}%</span>
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
          <span className="text-dark-muted">
            Негативные <span className="text-red-400 font-medium">{negPercent.toFixed(0)}%</span>
          </span>
        </div>
      </div>
    </div>
  );
}

export default SentimentBar;
