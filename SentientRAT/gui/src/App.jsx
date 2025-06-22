import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar.jsx';
import Chat from './Chat.jsx';
import ModelManager from './ModelManager.jsx';
import Canvas from './Canvas.jsx';
import ThemeToggle from './ThemeToggle.jsx';

const App = () => {
    const [activeTab, setActiveTab] = useState('chat');
    const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

    useEffect(() => {
        localStorage.setItem('theme', theme);
        document.body.className = theme;
    }, [theme]);

    return (
        <div className={`min-h-screen ${theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'} flex`}>
            <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
            <div className="flex-grow flex flex-col">
                <header className={`p-4 flex justify-between items-center ${theme === 'dark' ? 'bg-gray-800' : 'bg-white shadow'}`}>
                    <h1 className="text-2xl font-bold">SentientRAT</h1>
                    <ThemeToggle theme={theme} setTheme={setTheme} />
                </header>
                <main className="flex-grow p-6">
                    {activeTab === 'chat' && <Chat theme={theme} />}
                    {activeTab === 'models' && <ModelManager theme={theme} />}
                    {activeTab === 'canvas' && <Canvas theme={theme} />}
                </main>
            </div>
        </div>
    );
};

export default App;