import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'https://cdn.jsdelivr.net/npm/framer-motion@10.12.16/dist/framer-motion.min.js';

const Chat = ({ theme }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatRef = useRef(null);

    useEffect(() => {
        chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage = { sender: 'user', text: input };
        setMessages([...messages, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/process_command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer dummy_token'
                },
                body: JSON.stringify({ user_input: input })
            });
            const data = await response.json();
            const botMessage = { sender: 'bot', text: JSON.stringify(data, null, 2) };
            if (data.canvas) botMessage.canvas = data.canvas;
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            setMessages(prev => [...prev, { sender: 'bot', text: `Error: ${error.message}` }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className={`rounded-lg shadow-lg p-6 ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}
        >
            <div
                ref={chatRef}
                className="h-[70vh] overflow-y-auto mb-4 p-4 rounded-lg border"
            >
                {messages.map((msg, index) => (
                    <motion.div
                        key={index}
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ duration: 0.3 }}
                        className={`mb-4 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
                    >
                        <div
                            className={`inline-block p-3 rounded-2xl max-w-xs ${msg.sender === 'user' ? 'bg-blue-600 text-white' : theme === 'dark' ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-900'}`}
                        >
                            <pre className="whitespace-pre-wrap">{msg.text}</pre>
                            {msg.canvas && <div dangerouslySetInnerHTML={{ __html: msg.canvas }} />}
                        </div>
                    </motion.div>
                ))}
                {isLoading && (
                    <div className="text-left">
                        <div className="inline-block p-3 rounded-2xl bg-gray-200">
                            <span className="animate-pulse">Typing...</span>
                        </div>
                    </div>
                )}
            </div>
            <div className="flex">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    className={`flex-grow p-3 rounded-l-2xl border ${theme === 'dark' ? 'bg-gray-700 text-white border-gray-600' : 'bg-white text-gray-900 border-gray-300'}`}
                    placeholder="Type your command..."
                    disabled={isLoading}
                />
                <button
                    onClick={sendMessage}
                    className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-3 rounded-r-2xl"
                    disabled={isLoading}
                >
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </motion.div>
    );
};

export default Chat;