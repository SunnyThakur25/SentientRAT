import React from 'react';
import { motion } from 'https://cdn.jsdelivr.net/npm/framer-motion@10.12.16/dist/framer-motion.min.js';

const Canvas = ({ theme }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className={`rounded-lg shadow-lg p-6 ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}
        >
            <h2 className="text-2xl font-bold mb-6">Visualization</h2>
            <p className="text-gray-500">Run a scan command in the Chat tab to display interactive visualizations here.</p>
        </motion.div>
    );
};

export default Canvas;