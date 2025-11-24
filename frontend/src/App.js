import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Search, Settings, Clock, FileText, ChevronDown, ChevronUp, X, Loader2, History, Sparkles } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [stats, setStats] = useState({ total_snippets: 0 });
  const [showSearchHistory, setShowSearchHistory] = useState(false);
  const [searchHistoryRef, setSearchHistoryRef] = useState(null);
  
  // Settings
  const [topK, setTopK] = useState(3);
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.0);
  const [modelName, setModelName] = useState('gemini-2.5-flash');
  const [preferSameLang, setPreferSameLang] = useState(true);
  const [strictSameLang, setStrictSameLang] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  // Expanded items
  const [expandedItems, setExpandedItems] = useState({});

  useEffect(() => {
    loadStats();
    const savedHistory = localStorage.getItem('legalRagHistory');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        setHistory(parsed);
        if (parsed.length > 0) {
          setSelectedItem(parsed[0]);
        }
      } catch (e) {
        console.error('Error loading history:', e);
      }
    }
  }, []);

  // Close search history when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchHistoryRef && !searchHistoryRef.contains(event.target)) {
        setShowSearchHistory(false);
      }
    };

    if (showSearchHistory) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showSearchHistory, searchHistoryRef]);

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const saveHistory = (newHistory) => {
    setHistory(newHistory);
    localStorage.setItem('legalRagHistory', JSON.stringify(newHistory));
  };

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/query`, {
        query: query.trim(),
        top_k: topK,
        max_tokens: maxTokens,
        temperature: temperature,
        model_name: modelName,
        prefer_same_language: preferSameLang,
        strict_same_language: strictSameLang,
        include_prompt: true
      });

      const newItem = {
        id: Date.now(),
        query: query.trim(),
        answer: response.data.answer,
        retrieved: response.data.retrieved,
        prompt: response.data.prompt,
        total_time: response.data.total_time,
        retrieval_time: response.data.retrieval_time,
        generation_time: response.data.generation_time,
        query_lang: response.data.query_lang,
        timestamp: new Date().toISOString()
      };

      const newHistory = [newItem, ...history];
      saveHistory(newHistory);
      setSelectedItem(newItem);
      setQuery('');
    } catch (error) {
      alert(`خطأ: ${error.response?.data?.detail || error.message}`);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('هل أنت متأكد من حذف جميع السجلات؟')) {
      saveHistory([]);
      setExpandedItems({});
      setSelectedItem(null);
    }
  };

  const toggleExpanded = (itemId, type) => {
    const key = `${itemId}-${type}`;
    setExpandedItems(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const formatTime = (seconds) => {
    return seconds.toFixed(2) + 's';
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('ar-EG', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Get unique search queries from history
  const getSearchHistory = () => {
    const uniqueQueries = [];
    const seen = new Set();
    
    history.forEach(item => {
      const query = item.query.trim();
      if (query && !seen.has(query.toLowerCase())) {
        seen.add(query.toLowerCase());
        uniqueQueries.push({
          query: query,
          timestamp: item.timestamp,
          lang: item.query_lang
        });
      }
    });
    
    return uniqueQueries;
  };

  // Filter search history based on current query
  const filteredSearchHistory = () => {
    const allHistory = getSearchHistory();
    if (!query.trim()) {
      return allHistory.slice(0, 15); // Show last 15 if no query
    }
    return allHistory.filter(item => 
      item.query.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 15);
  };

  const handleQuerySelect = (selectedQuery) => {
    setQuery(selectedQuery);
    setShowSearchHistory(false);
    // Find the item in history and select it
    const item = history.find(h => h.query === selectedQuery);
    if (item) {
      setSelectedItem(item);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>M-law</h1>
          <p className="subtitle">🔧 القانون بين يديك </p>
        </div>
        <button 
          className="settings-toggle"
          onClick={() => setShowSettings(!showSettings)}
        >
          <Settings size={20} />
        </button>
      </header>

      <div className="main-container">
        {/* Left Sidebar - History */}
        <aside className="history-sidebar">
          <div className="history-sidebar-header">
            <History size={20} />
            <h2>📝 السجل</h2>
            {history.length > 0 && (
              <button className="clear-history-btn" onClick={clearHistory} title="مسح السجل">
                <X size={16} />
              </button>
            )}
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <div className="empty-history">
                <p>لا توجد أسئلة سابقة</p>
                <p className="empty-hint">ابدأ بكتابة سؤال جديد</p>
              </div>
            ) : (
              history.map((item) => (
                <div
                  key={item.id}
                  className={`history-item-card ${selectedItem?.id === item.id ? 'active' : ''}`}
                  onClick={() => setSelectedItem(item)}
                >
                  <div className="history-item-header">
                    <div className="history-item-icon">
                      <Sparkles size={14} />
                    </div>
                    <div className="history-item-time">{formatDate(item.timestamp)}</div>
                  </div>
                  <div className="history-item-query">{item.query}</div>
                  <div className="history-item-meta">
                    <span className="history-lang-badge">{item.query_lang === 'ar' ? 'عربي' : item.query_lang === 'fr' ? 'Français' : 'Other'}</span>
                    <span className="history-time-badge">{formatTime(item.total_time)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="main-content-area">
          {/* Query Input */}
          <div className="query-section">
            <div className="query-input-wrapper" ref={setSearchHistoryRef}>
              <div className="query-input-container">
                <input
                  type="text"
                  className="query-input"
                  placeholder="اكتب هنا / Écrivez ici"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setShowSearchHistory(true);
                  }}
                  onFocus={() => setShowSearchHistory(true)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !loading) {
                      setShowSearchHistory(false);
                      handleQuery();
                    }
                  }}
                  disabled={loading}
                />
                <button
                  className="query-button"
                  onClick={() => {
                    setShowSearchHistory(false);
                    handleQuery();
                  }}
                  disabled={loading || !query.trim()}
                >
                  {loading ? <Loader2 className="spinner" size={20} /> : <Search size={20} />}
                  <span>🔍 اسأل</span>
                </button>
              </div>
              
              {/* Search History Dropdown */}
              {showSearchHistory && filteredSearchHistory().length > 0 && (
                <div className="search-history-dropdown">
                  <div className="search-history-header">
                    <History size={16} />
                    <span>تاريخ البحث ({filteredSearchHistory().length})</span>
                  </div>
                  <div className="search-history-list">
                    {filteredSearchHistory().map((item, idx) => (
                      <div
                        key={idx}
                        className="search-history-item"
                        onClick={() => handleQuerySelect(item.query)}
                      >
                        <div className="search-history-query">
                          <Search size={14} />
                          <span>{item.query}</span>
                        </div>
                        <div className="search-history-meta">
                          <span className="search-history-lang">
                            {item.lang === 'ar' ? 'عربي' : item.lang === 'fr' ? 'Français' : 'Other'}
                          </span>
                          <span className="search-history-time">
                            {formatDate(item.timestamp)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Response Display */}
          {loading ? (
            <div className="loading-state">
              <Loader2 className="spinner-large" size={48} />
              <p>جاري البحث والتوليد...</p>
            </div>
          ) : selectedItem ? (
            <div className="response-container">
              <div className="response-header">
                <div className="response-title">
                  <h2>❓ {selectedItem.query}</h2>
                  <div className="response-meta">
                    <span className="response-time">{formatDate(selectedItem.timestamp)}</span>
                    <span className="response-lang">{selectedItem.query_lang === 'ar' ? 'عربي' : selectedItem.query_lang === 'fr' ? 'Français' : 'Other'}</span>
                  </div>
                </div>
                <div className="time-metrics">
                  <div className="metric-card">
                    <Clock size={16} />
                    <div>
                      <div className="metric-label">إجمالي</div>
                      <div className="metric-value">{formatTime(selectedItem.total_time)}</div>
                    </div>
                  </div>
                  <div className="metric-card">
                    <Search size={16} />
                    <div>
                      <div className="metric-label">استرجاع</div>
                      <div className="metric-value">{formatTime(selectedItem.retrieval_time)}</div>
                    </div>
                  </div>
                  <div className="metric-card">
                    <Sparkles size={16} />
                    <div>
                      <div className="metric-label">توليد</div>
                      <div className="metric-value">{formatTime(selectedItem.generation_time)}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="answer-box">
                <div className="answer-content">
                  {selectedItem.answer.split('\n').map((line, idx) => (
                    <p key={idx}>{line || <br />}</p>
                  ))}
                </div>
              </div>

              {/* Prompt Toggle */}
              {selectedItem.prompt && (
                <div className="prompt-section">
                  <button
                    className="toggle-button"
                    onClick={() => toggleExpanded(selectedItem.id, 'prompt')}
                  >
                    {expandedItems[`${selectedItem.id}-prompt`] ? (
                      <ChevronUp size={16} />
                    ) : (
                      <ChevronDown size={16} />
                    )}
                    <span>{expandedItems[`${selectedItem.id}-prompt`] ? 'إخفاء' : 'إظهار'} الطلب (Prompt)</span>
                  </button>
                  {expandedItems[`${selectedItem.id}-prompt`] && (
                    <div className="prompt-content">
                      <pre>{selectedItem.prompt}</pre>
                    </div>
                  )}
                </div>
              )}

              {/* Resources */}
              <div className="resources-section">
                <button
                  className="toggle-button"
                  onClick={() => toggleExpanded(selectedItem.id, 'resources')}
                >
                  {expandedItems[`${selectedItem.id}-resources`] ? (
                    <ChevronUp size={16} />
                  ) : (
                    <ChevronDown size={16} />
                  )}
                  <FileText size={16} />
                  <span>📚 المصادر المستخدمة ({selectedItem.retrieved?.length || 0})</span>
                </button>
                {expandedItems[`${selectedItem.id}-resources`] && (
                  <div className="resources-list">
                    {selectedItem.retrieved?.map((r, idx) => {
                      const meta = r.meta || {};
                      return (
                        <div key={idx} className="resource-item">
                          <div className="resource-header">
                            <strong>
                              ({meta.mada || ''} : {meta.bab || ''} : {meta.source || meta.bab || ''})
                            </strong>
                            <div className="resource-meta">
                              <span className="lang-badge">لغة: {meta.lang || 'other'}</span>
                              <span className="score-badge">تشابه: {r.score?.toFixed(3) || '0.000'}</span>
                            </div>
                          </div>
                          <div className="resource-text">
                            {r.text || ''}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">⚖️</div>
              <h2>مرحباً بك في محامي افتراضي</h2>
              <p>👆 اكتب سؤالك القانوني أعلاه للبدء</p>
              <div className="empty-state-stats">
                <div className="stat-card">
                  <strong>{stats.total_snippets.toLocaleString()}</strong>
                  <span>مقتطف قانوني متاح</span>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Right Sidebar - Settings */}
        <aside className={`settings-sidebar ${showSettings ? 'open' : ''}`}>
          <div className="sidebar-header">
            <h2>⚙️ الإعدادات</h2>
            <button className="close-btn" onClick={() => setShowSettings(false)}>
              <X size={20} />
            </button>
          </div>

          <div className="settings-section">
            <label>
              عدد المقتطفات المسترجعة: {topK}
              <input
                type="range"
                min="1"
                max="8"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
              />
            </label>

            <label>
              طول الإجابة (حد أقصى): {maxTokens}
              <input
                type="range"
                min="128"
                max="2048"
                step="64"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              />
            </label>

            <label>
              درجة العشوائية (temperature): {temperature.toFixed(2)}
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
              />
            </label>

            <label>
              نموذج التوليد (Gemini)
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="gemini-2.5-flash">gemini-2.5-flash</option>
              </select>
            </label>
          </div>

          <div className="settings-section">
            <h3>خيارات التحكم في اللغة</h3>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={preferSameLang}
                  onChange={(e) => setPreferSameLang(e.target.checked)}
                />
                فضّل مقتطفات بنفس لغة السؤال (افتراضي)
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={strictSameLang}
                  onChange={(e) => setStrictSameLang(e.target.checked)}
                />
                التصفية الصارمة: إرجاع مقتطفات من نفس اللغة فقط
              </label>
            </div>
          </div>

          <div className="settings-section">
            <h3>📊 الإحصائيات</h3>
            <div className="stat-item">
              <span>المقتطفات المتاحة:</span>
              <strong>{stats.total_snippets.toLocaleString()}</strong>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
