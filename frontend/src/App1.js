import React, { useState } from 'react';
import './App1.css';
import logo from './Logo.png';

const API_URL = 'http://127.0.0.1:5000/checkSMS';

export default function App1() {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);         
  const [finalPrediction, setFinalPrediction] = useState(null); 
  const [suspiciousWords, setSuspiciousWords] = useState([]);  
  const [messageInfo, setMessageInfo] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setResults(null);
    setFinalPrediction(null);
    setSuspiciousWords([]);
    setMessageInfo('');
    setError(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        let serverMsg = '';
        try {
          const maybeJson = await response.json();
          serverMsg = maybeJson?.error || '';
        } catch (_) {}
        throw new Error(serverMsg || `Server error: ${response.status}`);
      }

      const data = await response.json();
      if (!data || Object.keys(data).length === 0) {
        throw new Error('No results received from server.');
      }

      setResults(data.results || null);
      setFinalPrediction(
        typeof data.final_prediction === 'number' ? data.final_prediction : null
      );
      setSuspiciousWords(Array.isArray(data.suspicious_words) ? data.suspicious_words : []);
      setMessageInfo(data.message_info || '');

    } catch (err) {
      console.error('Error fetching predictions:', err);
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const verdictText =
    finalPrediction === 1 ? 'Phishing (Suspicious)' :
    finalPrediction === 0 ? 'Safe' : '';

  return (
    <div className="App">
      <header className="App-header">
        <img
          src={logo}
          alt="Check Phishing SMS"
          className="app-logo"
        />
      </header>

      <main>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to analyze..."
        />

        <button onClick={handlePredict} disabled={!text.trim() || loading}>
          {loading ? 'Analyzing...' : 'Analyze SMS'}
        </button>

        {loading && <p className="loading">Analyzing, please wait...</p>}
        {error && <p className="error-message">{error}</p>}

        {finalPrediction !== null && (
          <div className="final-card">
            <div
              className={
                finalPrediction === 1 ? 'final-badge suspicious' : 'final-badge safe'
              }
            >
              {verdictText}
            </div>

            {finalPrediction === 1 && suspiciousWords?.length > 0 && (
              <div className="suspicious-words">
                <h3>Top 5 Suspicious Words</h3>
                <ul>
                  {suspiciousWords.map((word, index) => {
                    return (
                      <li key={index} className="suspicious-word-item">
                        {word}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        )}
        {results && (
          <div className="result-container">
            <h2>Models</h2>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Prediction</th>
                  <th>Time (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(results).map((model) => (
                  <tr key={model}>
                    <td>{model}</td>
                    <td
                      className={
                        results[model].prediction === 1 ? 'suspicious' : 'safe'
                      }
                    >
                      {results[model].prediction === 1 ? 'Suspicious' : 'Safe'}
                    </td>
                    <td>{results[model].time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {messageInfo && <p className="message-info">{messageInfo}</p>}
          </div>
        )}
      </main>
    </div>
  );
}
