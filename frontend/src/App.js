import React, { useState } from 'react';
import './App.css';
import logo from './Logo.png';

function App() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [suspiciousWords, setSuspiciousWords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputError, setInputError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (message.trim() === '') {
      setInputError("The message is empty");
      setResult(null);
      setSuspiciousWords([]);
      setError(null);
      return;
    }
    setInputError('');
    setLoading(true);
    setResult(null);
    setSuspiciousWords([]);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        throw new Error(`ERROR HTTP!: ${response.status}`);
      }
      const data = await response.json();
      setResult(data.result);
      setSuspiciousWords(data.suspicious_words);
    }
    catch (err) {
      console.error("Error checking message", err);
      setError("Connection to server failed.");
    } finally {
      setLoading(false);
    }
  };

  let resultSection = null;

  if (result !== null) {
    let resultMessage = null;
    if (result === 1) {
      resultMessage = <p className="suspicious">fishing message.</p>;
    } else {
      resultMessage = <p className="safe">Reliable message</p>;
    }
    let suspiciousWordsSection = null;
    
    if (suspiciousWords.length > 0) {
      const wordItems = suspiciousWords.map((word, index) => {
        return <li key={index}>{word}</li>;
      });
      suspiciousWordsSection = (
        <div className="suspicious-words">
          <h3>Suspicious words:</h3>
          <ul>{wordItems}</ul>
        </div>
      );
    }

    resultSection = (
      <div className="result-container">
        <h2>Result:</h2>
        {resultMessage}
        {suspiciousWordsSection}
      </div>
    );
  }

  let errorMessage = null;
  if (error !== null) {
    errorMessage = <p className="error-message">{error}</p>;
  }

  let inputErrorMessage = null;
  if (inputError !== '') {
    inputErrorMessage = <p className="input-error-message">{inputError}</p>;
  }

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
        <form onSubmit={handleSubmit}>
          <textarea
            placeholder="Enter the SMS message you received for verification."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows="5"
            cols="50"
          ></textarea>
          <br />
          {inputErrorMessage}
          <button type="submit" disabled={loading}>
            {loading ? 'Please wait, we are checking.' : 'Check the message'}
          </button>
        </form>

        {errorMessage}
        {resultSection}
      </main>
    </div>
  );
}

export default App;
