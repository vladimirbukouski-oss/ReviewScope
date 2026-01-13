/**
 * Main App Component with routing
 */

import { Routes, Route } from 'react-router-dom';
import { Header } from './components';
import { HomePage, AnalysisPage, HistoryPage } from './pages';

function App() {
  return (
    <div className="min-h-screen bg-dark-bg text-dark-text">
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/analysis/:sessionId" element={<AnalysisPage />} />
          <Route path="/history" element={<HistoryPage />} />
          {/* Future routes */}
          {/* <Route path="/compare" element={<ComparePage />} /> */}
        </Routes>
      </main>
    </div>
  );
}

export default App;
