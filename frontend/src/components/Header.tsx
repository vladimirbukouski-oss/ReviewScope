/**
 * Header Component
 */

import { Link } from 'react-router-dom';

export function Header() {
  return (
    <header className="sticky top-0 z-50 glass border-b border-dark-border">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">Отзывоскоп</h1>
            <p className="text-xs text-dark-muted">AI-анализ отзывов</p>
          </div>
        </Link>

        {/* Service badges */}
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 text-xs font-medium bg-purple-500/20 text-purple-400 rounded-full">
            WB
          </span>
          <span className="px-3 py-1 text-xs font-medium bg-orange-500/20 text-orange-400 rounded-full">
            Onliner
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
