import React, { useState } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [query, setQuery] = useState('');
  const [chats, setChats] = useState([]);
  const remoline=(text)=>{
    const tex=text;
    const finaltext=tex.split('\n').map(str => <p>{str}</p>);;
    return finaltext
  }
  const handleQuery = async () => {
    if (query.trim()) {
      try {
        const response = await fetch(`http://127.0.0.1:8000/ds/${query}`, {
          method: 'GET',
          
        });
  
        const data = await response.json();
        const newChat = { query, response: remoline(data.response), codeResponse: data.codeResponse };
        
        setChats([newChat, ...chats]);
        setQuery(''); // Clear the query input field
      } catch (error) {
        console.error('Error fetching response:', error);
      }
    }
  };
  

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleQuery();
    }
  };

  const startNewChat = () => {
    setQuery('');
    setChats([]); // Clear chat history when starting a new chat
  };

  return (
    <div className="chat-interface">
      <div className="chat-area">
        <div className="chat-content">
          <h2>Welcome to the Chat</h2>
          {chats.map((chat, index) => (
            <div key={index} className="chat-item">
              <div className="response-section">
                <p><strong>Answer:</strong> {chat.response}</p>
              </div>
              <div className="code-section">
                <p><strong>Code Response:</strong> {chat.codeResponse}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="qa-section">
          <input
            type="text"
            placeholder="Ask your question here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button onClick={handleQuery}>Submit</button>
        </div>
      </div>
      
      <div className="sidebar">
        <button className="new-chat-btn" onClick={startNewChat}>
          New Chat
        </button>
        <div className="chat-history">
          {chats.length === 0 ? <p>No history yet</p> : 
            chats.map((chat, index) => (
              <div key={index} className="history-item">
                {chat.query}
              </div>
            ))
          }
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
