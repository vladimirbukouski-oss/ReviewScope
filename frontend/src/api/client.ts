/**
 * ReviewScope API Client
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';

export interface AnalysisStatus {
  session_id: string;
  status: 'pending' | 'fetching' | 'scoring' | 'building_rag' | 'summarizing' | 'ready' | 'error';
  progress: number;
  message: string;
  eta_seconds?: number;
}

export interface Product {
  url: string;
  service: 'wb' | 'onliner';
  truthStars: number;
  avgOrig: number;
  suspiciousShare: number;
  totalReviews: number;
  keptReviews: number;
  sentimentMix: {
    neg: number;
    neu: number;
    pos: number;
  };
}

export interface Summary {
  one_liner: string;
  score: {
    value: number;
    scale: number;
    note: string;
  };
  pros: Array<{ point: string; evidence_ids: string[] }>;
  cons: Array<{ point: string; evidence_ids: string[] }>;
  aspects: Array<{
    aspect: string;
    summary: string;
    mentions: number;
    evidence_ids: string[];
  }>;
  who_it_is_for: string[];
  who_should_avoid: string[];
  confidence: {
    level: 'low' | 'medium' | 'high';
    why: string;
  };
}

export interface Review {
  id: string;
  user: string;
  rating: number;
  trust: number;
  text: string;
  created: string;
  sentiment: 'pos' | 'neu' | 'neg';
  pred_star_soft?: number;
}

export interface ChatMessage {
  answer: string;
  evidence: Review[];
}

export interface RecentAnalysis {
  session_id: string;
  url: string;
  service: string;
  truth_stars: number;
  total_reviews: number;
  product_name: string;
  created_at: string;
}

class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Analysis endpoints
  async analyze(url: string, useCache = true): Promise<AnalysisStatus> {
    return this.request<AnalysisStatus>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ url, use_cache: useCache }),
    });
  }

  async getStatus(sessionId: string): Promise<AnalysisStatus> {
    return this.request<AnalysisStatus>(`/status/${sessionId}`);
  }

  async getSummary(sessionId: string): Promise<{
    session_id: string;
    product: Product;
    summary: Summary;
    reviews: Review[];
  }> {
    return this.request(`/summary/${sessionId}`);
  }

  // Chat endpoint
  async chat(sessionId: string, question: string): Promise<ChatMessage> {
    return this.request<ChatMessage>(`/chat/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, question }),
    });
  }

  // Reviews endpoint
  async getReviews(
    sessionId: string,
    skip = 0,
    limit = 20,
    sortBy = 'trust'
  ): Promise<{
    total: number;
    skip: number;
    limit: number;
    reviews: Review[];
  }> {
    return this.request(
      `/reviews/${sessionId}?skip=${skip}&limit=${limit}&sort_by=${sortBy}`
    );
  }

  // Recent analyses
  async getRecentAnalyses(limit = 20, offset = 0): Promise<{
    total: number;
    analyses: RecentAnalysis[];
  }> {
    return this.request(`/analyses/recent?limit=${limit}&offset=${offset}`);
  }

  // Analysis metadata (for sharing)
  async getAnalysisMeta(sessionId: string): Promise<{
    session_id: string;
    url: string;
    service: string;
    product_name: string;
    truth_stars: number;
    total_reviews: number;
    one_liner: string;
    score: { value: number; scale: number };
  }> {
    return this.request(`/analysis/${sessionId}/meta`);
  }

  // Health check
  async health(): Promise<{
    status: string;
    active_sessions: number;
    config: Record<string, string>;
  }> {
    return this.request('/health');
  }
}

export const api = new APIClient();
export default api;
