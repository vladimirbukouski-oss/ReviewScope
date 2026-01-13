/**
 * Custom hook for chat functionality
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { api, type Review } from '../api/client';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  evidence?: Review[];
  timestamp: Date;
}

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message
  const sendMessage = useCallback(
    async (question: string) => {
      if (!sessionId || !question.trim()) return;

      // Add user message
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        type: 'user',
        content: question.trim(),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsTyping(true);
      setError(null);

      try {
        const response = await api.chat(sessionId, question);

        // Add assistant message
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: response.answer,
          evidence: response.evidence,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Ошибка отправки сообщения');
      } finally {
        setIsTyping(false);
      }
    },
    [sessionId]
  );

  // Clear chat
  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isTyping,
    error,
    sendMessage,
    clearChat,
    messagesEndRef,
  };
}

export default useChat;
