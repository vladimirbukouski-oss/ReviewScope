/**
 * Chat Section Component
 */

import { useState, type FormEvent } from 'react';
import { useChat } from '../hooks/useChat';
import { ReviewCard } from './ReviewCard';
import clsx from 'clsx';

interface ChatSectionProps {
  sessionId: string;
}

const QUICK_QUESTIONS = [
  { label: '👍 Плюсы', question: 'Какие основные плюсы этого товара?' },
  { label: '👎 Минусы', question: 'Какие основные минусы и недостатки?' },
  { label: '💰 Цена/качество', question: 'Стоит ли товар своих денег?' },
  { label: '📦 Доставка', question: 'Как отзываются о доставке и упаковке?' },
];

export function ChatSection({ sessionId }: ChatSectionProps) {
  const [input, setInput] = useState('');
  const { messages, isTyping, error, sendMessage, messagesEndRef } = useChat(sessionId);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      sendMessage(input);
      setInput('');
    }
  };

  const handleQuickQuestion = (question: string) => {
    sendMessage(question);
  };

  return (
    <div className="flex flex-col h-[600px]">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary-500/20 flex items-center justify-center">
              <span className="text-3xl">🤖</span>
            </div>
            <h3 className="text-lg font-medium mb-2">Задайте вопрос об отзывах</h3>
            <p className="text-dark-muted text-sm mb-6">
              AI проанализирует отзывы и даст ответ с доказательствами
            </p>

            {/* Quick questions */}
            <div className="flex flex-wrap justify-center gap-2">
              {QUICK_QUESTIONS.map((q) => (
                <button
                  key={q.label}
                  onClick={() => handleQuickQuestion(q.question)}
                  className="px-4 py-2 text-sm bg-dark-card border border-dark-border rounded-full hover:border-primary-500 transition-colors"
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={clsx(
                  'max-w-[85%] animate-slide-up',
                  msg.type === 'user' ? 'ml-auto' : 'mr-auto'
                )}
              >
                {/* Message bubble */}
                <div
                  className={clsx(
                    'rounded-2xl px-4 py-3',
                    msg.type === 'user'
                      ? 'bg-primary-600 text-white rounded-br-md'
                      : 'bg-dark-card border border-dark-border rounded-bl-md'
                  )}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>

                {/* Evidence */}
                {msg.evidence && msg.evidence.length > 0 && (
                  <div className="mt-3 space-y-2">
                    <p className="text-xs text-dark-muted ml-1">Доказательства:</p>
                    {msg.evidence.slice(0, 3).map((review) => (
                      <ReviewCard key={review.id} review={review} isEvidence className="text-sm" />
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {isTyping && (
              <div className="flex items-center gap-2 text-dark-muted">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm">AI печатает...</span>
              </div>
            )}
          </>
        )}

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-500/20 text-red-400 rounded-lg text-sm">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-dark-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Задайте вопрос об отзывах..."
            className="flex-1 px-4 py-3 bg-dark-card border border-dark-border rounded-xl focus:outline-none focus:border-primary-500 transition-colors"
            disabled={isTyping}
          />
          <button
            type="submit"
            disabled={!input.trim() || isTyping}
            className="px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}

export default ChatSection;
