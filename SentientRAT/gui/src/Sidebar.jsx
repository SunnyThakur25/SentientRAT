import React from 'react';
import { motion } from 'https://cdn.jsdelivr.net/npm/framer-motion@10.12.16/dist/framer-motion.min.js';

const Sidebar = ({ activeTab, setActiveTab }) => {
    return (
        <motion.nav
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            transition={{ duration: 0.5 }}
            className="w-64 bg-gradient-to-b from-blue-600 to-blue-800 text-white p-6 flex flex-col"
        >
            <h2 className="text-xl font-bold mb-8">Menu</h2>
            <button
                className={`mb-4 text-left ${activeTab === 'chat' ? 'font-bold underline' : ''}`}
                onClick={() => setActiveTab('chat')}
            >
                Chat
            </button>
            <button
                className={`mb-4 text-left ${activeTab === 'models' ? 'font-bold underline' : ''}`}
                onClick={() => setActiveTab('models')}
            >
                Model Manager
            </button>
            <button
                className={`mb-4 text-left ${activeTab === 'canvas' ? 'font-bold underline' : ''}`}
                onClick={() => setActiveTab('canvas')}
            >
                Visualization
            </button>
        </motion.nav>
    );
};

export default Sidebar;