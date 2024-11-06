import React, { useState } from 'react';

const Home = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleQuery = async () => {
    if (query.trim()) {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query }),
        });

        const data = await res.json();
        setResponse(data.response);  
      } catch (error) {
        console.error('Error fetching response:', error);
      }
    }
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
