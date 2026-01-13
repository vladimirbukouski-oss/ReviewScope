/**
 * Custom hook for analysis operations
 */

import { useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useAnalysisStore } from '../store/analysisStore';

const POLL_INTERVAL = 2000; // 2 seconds

export function useAnalysis() {
  const navigate = useNavigate();
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const {
    sessionId,
    status,
    product,
    summary,
    reviews,
    isLoading,
    error,
    setSessionId,
    setStatus,
    setProduct,
    setSummary,
    setReviews,
    setLoading,
    setError,
    addToHistory,
    reset,
  } = useAnalysisStore();

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Fetch summary when ready
  const fetchSummary = useCallback(
    async (sid: string) => {
      try {
        const data = await api.getSummary(sid);
        setProduct(data.product);
        setSummary(data.summary);
        setReviews(data.reviews);

        // Add to history
        addToHistory({
          sessionId: sid,
          url: data.product.url,
          service: data.product.service,
          productName: data.summary.one_liner.slice(0, 100),
          truthStars: data.product.truthStars,
          totalReviews: data.product.totalReviews,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Ошибка загрузки данных');
      }
    },
    [setProduct, setSummary, setReviews, setError, addToHistory]
  );

  // Poll status
  const pollStatus = useCallback(
    async (sid: string) => {
      try {
        const statusData = await api.getStatus(sid);
        setStatus(statusData);

        if (statusData.status === 'ready') {
          stopPolling();
          setLoading(false);
          await fetchSummary(sid);
        } else if (statusData.status === 'error') {
          stopPolling();
          setLoading(false);
          setError(statusData.message);
        }
      } catch (err) {
        stopPolling();
        setLoading(false);
        setError(err instanceof Error ? err.message : 'Ошибка проверки статуса');
      }
    },
    [setStatus, setLoading, setError, stopPolling, fetchSummary]
  );

  // Start analysis
  const startAnalysis = useCallback(
    async (url: string, useCache = true) => {
      reset();
      setLoading(true);
      setError(null);

      try {
        const result = await api.analyze(url, useCache);
        setSessionId(result.session_id);
        setStatus(result);

        if (result.status === 'ready') {
          // Cached result
          setLoading(false);
          await fetchSummary(result.session_id);
          navigate(`/analysis/${result.session_id}`);
        } else {
          // Start polling
          navigate(`/analysis/${result.session_id}`);
          pollRef.current = setInterval(
            () => pollStatus(result.session_id),
            POLL_INTERVAL
          );
        }
      } catch (err) {
        setLoading(false);
        setError(err instanceof Error ? err.message : 'Ошибка запуска анализа');
      }
    },
    [reset, setLoading, setError, setSessionId, setStatus, navigate, fetchSummary, pollStatus]
  );

  // Load existing analysis
  const loadAnalysis = useCallback(
    async (sid: string) => {
      if (sessionId === sid && status?.status === 'ready') {
        return; // Already loaded
      }

      setSessionId(sid);
      setLoading(true);
      setError(null);

      try {
        const statusData = await api.getStatus(sid);
        setStatus(statusData);

        if (statusData.status === 'ready') {
          await fetchSummary(sid);
          setLoading(false);
        } else if (statusData.status === 'error') {
          setLoading(false);
          setError(statusData.message);
        } else {
          // Still processing, start polling
          pollRef.current = setInterval(() => pollStatus(sid), POLL_INTERVAL);
        }
      } catch (err) {
        setLoading(false);
        setError(err instanceof Error ? err.message : 'Ошибка загрузки анализа');
      }
    },
    [sessionId, status, setSessionId, setLoading, setError, setStatus, fetchSummary, pollStatus]
  );

  // Load more reviews
  const loadMoreReviews = useCallback(
    async (sortBy = 'trust') => {
      if (!sessionId) return;

      try {
        const data = await api.getReviews(sessionId, reviews.length, 20, sortBy);
        setReviews([...reviews, ...data.reviews]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Ошибка загрузки отзывов');
      }
    },
    [sessionId, reviews, setReviews, setError]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    sessionId,
    status,
    product,
    summary,
    reviews,
    isLoading,
    error,
    startAnalysis,
    loadAnalysis,
    loadMoreReviews,
    reset,
  };
}

export default useAnalysis;
