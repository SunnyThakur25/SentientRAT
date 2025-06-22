import React, { useState, useEffect } from 'react';
import { motion } from 'https://cdn.jsdelivr.net/npm/framer-motion@10.12.16/dist/framer-motion.min.js';

const ModelManager = ({ theme }) => {
    const [systemInfo, setSystemInfo] = useState(null);
    const [selectedModel, setSelectedModel] = useState('');
    const [downloadStatus, setDownloadStatus] = useState('');
    const [progress, setProgress] = useState(0);

    const models = [
        { name: 'MiniMax M1', id: 'minimax/m1', requires: 'GPU' },
        { name: 'DeepSeek R1', id: 'deepseek/r1', requires: 'GPU' },
        { name: 'LLaMA 3.1-8B', id: 'meta-llama/llama-3.1-8b', requires: 'CPU/GPU' }
    ];

    useEffect(() => {
        fetch('http://localhost:8000/system_info')
            .then(res => res.json())
            .then(data => setSystemInfo(data))
            .catch(err => setDownloadStatus(`Error fetching system info: ${err.message}`));
    }, []);

    const downloadModel = async () => {
        if (!selectedModel) return;
        setDownloadStatus('Downloading...');
        setProgress(0);

        // Simulate progress (replace with actual progress tracking if possible)
        const progressInterval = setInterval(() => {
            setProgress(prev => (prev < 90 ? prev + 10 : prev));
        }, 1000);

        try {
            const response = await fetch('http://localhost:8000/download_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer dummy_token'
                },
                body: JSON.stringify({ model_name: selectedModel })
            });
            const data = await response.json();
            clearInterval(progressInterval);
            setProgress(100);
            setDownloadStatus(data.status === 'success' ? 'Download complete!' : `Error: ${data.error}`);
        } catch (error) {
            clearInterval(progressInterval);
            setDownloadStatus(`Error: ${error.message}`);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className={`rounded-lg shadow-lg p-6 ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}
        >
            <h2 className="text-2xl font-bold mb-6">Model Manager</h2>
            {systemInfo && (
                <motion.div
                    initial={{ y: 20 }}
                    animate={{ y: 0 }}
                    className={`p-4 rounded-lg mb-6 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}
                >
                    <p><strong>OS:</strong> {systemInfo.os}</p>
                    <p><strong>CPU Cores:</strong> {systemInfo.cpu_cores}</p>
                    <p><strong>RAM:</strong> {systemInfo.ram_gb} GB</p>
                    <p><strong>GPU:</strong> {systemInfo.gpu_available ? 'Available' : 'Not Available'}</p>
                    <p><strong>Recommended:</strong> {systemInfo.recommended_models.join(', ')}</p>
                </motion.div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {models.map(model => (
                    <motion.div
                        key={model.id}
                        whileHover={{ scale: 1.05 }}
                        className={`p-4 rounded-lg shadow cursor-pointer ${selectedModel === model.id ? 'border-2 border-blue-600' : ''} ${theme === 'dark' ? 'bg-gray-700' : 'bg-white'}`}
                        onClick={() => setSelectedModel(model.id)}
                    >
                        <h3 className="font-bold">{model.name}</h3>
                        <p className="text-sm">{model.requires}</p>
                    </motion.div>
                ))}
            </div>
            <button
                onClick={downloadModel}
                className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-3 rounded-lg"
                disabled={!selectedModel || downloadStatus === 'Downloading...'}
            >
                Download Model
            </button>
            {downloadStatus && (
                <div className="mt-4">
                    <p className="text-sm">{downloadStatus}</p>
                    {downloadStatus === 'Downloading...' && (
                        <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                            <motion.div
                                className="bg-blue-600 h-2.5 rounded-full"
                                initial={{ width: '0%' }}
                                animate={{ width: `${progress}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                    )}
                </div>
            )}
        </motion.div>
    );
};

export default ModelManager;