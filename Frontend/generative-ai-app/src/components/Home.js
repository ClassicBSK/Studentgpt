import React, { useState } from 'react';

const Home = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleQuery = () => {
    // Placeholder for backend API call
    setResponse('This is the generated response to your question!');
  };

  return (
    <div className="home-container">
      <h2>Welcome to Generative AI</h2>
      <div className="qa-section">
        <input
          type="text"
          placeholder="Ask your question here..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleQuery}>Submit</button>
      </div>
      {response && (
        <div className="response-section">
          <h3>Response:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};

export default Home;
