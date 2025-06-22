import React from 'react';
import { motion } from 'https://cdn.jsdelivr.net/npm/framer-motion@10.12.16/dist/framer-motion.min.js';

const ThemeToggle = ({ theme, setTheme }) => {
    return (
        <motion.button
            whileHover={{ scale: 1.1 }}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-2 rounded-full bg-gradient-to-r from-blue-600 to-blue-800 text-white"
        >
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </motion.button>
    );
};

export default ThemeToggle;