// src/components/ChatInterface.js
import React, { useState } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [chats, setChats] = useState([]);

  const handleQuery = () => {
    // Placeholder for backend API call
    const newChat = { query, response: 'Generated response!' };
    setChats([newChat, ...chats]);
    setResponse('Generated response!');
    setQuery(''); // Clear the input field after submitting
  };

  const startNewChat = () => {
    setQuery('');
    setResponse('');
  };

  return (
    <div className="chat-interface">
      {/* Main chat area */}
      <div className="chat-area">
        <div className="chat-content">
          <h2>Welcome to the Chat</h2>
          {response && (
            <div className="response-section">
              <h3>Response:</h3>
              <p>{response}</p>
            </div>
          )}
        </div>
        
        {/* Input area at the bottom */}
        <div className="qa-section">
          <input
            type="text"
            placeholder="Ask your question here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={handleQuery}>Submit</button>
        </div>
      </div>

      {/* Sidebar */}
      <div className="sidebar">
        <button className="new-chat-btn" onClick={startNewChat}>
          New Chat
        </button>
        <div className="chat-history">
          {chats.length > 0 ? (
            chats.map((chat, index) => (
              <div key={index} className="chat-item">
                <p>{chat.query}</p>
              </div>
            ))
          ) : (
            <p>No history yet</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
