/**
 * Loading Overlay Component - shows during analysis
 */

import type { AnalysisStatus } from '../api/client';

interface LoadingOverlayProps {
  status: AnalysisStatus;
}

const STEPS = [
  { key: 'fetching', label: 'Сбор', icon: '📥' },
  { key: 'scoring', label: 'Анализ', icon: '🔍' },
  { key: 'building_rag', label: 'Индекс', icon: '📊' },
  { key: 'summarizing', label: 'Сводка', icon: '✨' },
];

export function LoadingOverlay({ status }: LoadingOverlayProps) {
  const currentStepIndex = STEPS.findIndex((s) => s.key === status.status);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-dark-bg/90 backdrop-blur-sm">
      <div className="max-w-md w-full mx-4 card p-8 text-center">
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

        {/* ETA */}
        {status.eta_seconds && status.eta_seconds > 0 && (
          <p className="text-sm text-dark-muted mt-2">
            Примерно {status.eta_seconds} сек.
          </p>
        )}
      </div>
    </div>
  );
}

export default LoadingOverlay;
