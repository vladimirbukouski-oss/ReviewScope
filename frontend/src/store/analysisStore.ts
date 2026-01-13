/**
 * Analysis State Management with Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Product, Summary, Review, AnalysisStatus } from '../api/client';

interface HistoryItem {
  sessionId: string;
  url: string;
  service: string;
  productName: string;
  truthStars: number;
  totalReviews: number;
  createdAt: string;
  isFavorite: boolean;
}

interface AnalysisState {
  // Current analysis
  sessionId: string | null;
  status: AnalysisStatus | null;
  product: Product | null;
  summary: Summary | null;
  reviews: Review[];
  isLoading: boolean;
  error: string | null;

  // UI state
  activeTab: 'summary' | 'reviews' | 'chat';

  // History (persisted in localStorage)
  history: HistoryItem[];

  // Actions
  setSessionId: (id: string | null) => void;
  setStatus: (status: AnalysisStatus | null) => void;
  setProduct: (product: Product | null) => void;
  setSummary: (summary: Summary | null) => void;
  setReviews: (reviews: Review[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: 'summary' | 'reviews' | 'chat') => void;

  // History actions
  addToHistory: (item: Omit<HistoryItem, 'isFavorite' | 'createdAt'>) => void;
  toggleFavorite: (sessionId: string) => void;
  removeFromHistory: (sessionId: string) => void;
  clearHistory: () => void;

  // Reset
  reset: () => void;
}

const MAX_HISTORY_ITEMS = 50;

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      // Initial state
      sessionId: null,
      status: null,
      product: null,
      summary: null,
      reviews: [],
      isLoading: false,
      error: null,
      activeTab: 'summary',
      history: [],

      // Setters
      setSessionId: (id) => set({ sessionId: id }),
      setStatus: (status) => set({ status }),
      setProduct: (product) => set({ product }),
      setSummary: (summary) => set({ summary }),
      setReviews: (reviews) => set({ reviews }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      setActiveTab: (activeTab) => set({ activeTab }),

      // History management
      addToHistory: (item) => {
        const { history } = get();
        const existingIndex = history.findIndex((h) => h.sessionId === item.sessionId);

        let newHistory: HistoryItem[];
        if (existingIndex >= 0) {
          // Move to top if exists
          const existing = history[existingIndex];
          newHistory = [
            { ...existing, ...item, createdAt: new Date().toISOString() },
            ...history.filter((_, i) => i !== existingIndex),
          ];
        } else {
          // Add new item
          newHistory = [
            { ...item, isFavorite: false, createdAt: new Date().toISOString() },
            ...history,
          ].slice(0, MAX_HISTORY_ITEMS);
        }

        set({ history: newHistory });
      },

      toggleFavorite: (sessionId) => {
        const { history } = get();
        set({
          history: history.map((h) =>
            h.sessionId === sessionId ? { ...h, isFavorite: !h.isFavorite } : h
          ),
        });
      },

      removeFromHistory: (sessionId) => {
        const { history } = get();
        set({ history: history.filter((h) => h.sessionId !== sessionId) });
      },

      clearHistory: () => set({ history: [] }),

      reset: () =>
        set({
          sessionId: null,
          status: null,
          product: null,
          summary: null,
          reviews: [],
          isLoading: false,
          error: null,
          activeTab: 'summary',
        }),
    }),
    {
      name: 'reviewscope-storage',
      partialize: (state) => ({ history: state.history }),
    }
  )
);

export default useAnalysisStore;
