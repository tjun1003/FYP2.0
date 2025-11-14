/**
 * Node.js Express Server (Refactored)
 * Calls the persistent FastAPI inference service instead of starting a Python process every time
 * 
 * Architecture Change:
 * Old: Frontend -> Node.js -> Start Python Process -> Load Model -> Predict -> Close Process
 * New: Frontend -> Node.js -> HTTP Request -> FastAPI Service (Model Resident) -> Predict -> Return
 */
/**
 * Node.js Express Server (RAG Enhanced)
 * Supports Retrieval-Augmented Generation functionality
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '..')));

// Configuration
const CONFIG = {
    inferenceServerUrl: process.env.INFERENCE_SERVER_URL || 'http://localhost:8000',
    inferenceServerTimeout: 60000, // Increased to 60 seconds to support web retrieval
    autoStartInference: (process.env.AUTO_START_INFERENCE || 'true').toLowerCase() !== 'false',
    inferenceCmd: process.env.INFERENCE_CMD || 'python',
    inferenceArgs: process.env.INFERENCE_ARGS ? process.env.INFERENCE_ARGS.split(' ') : ['-u', 'Inference.py'],
    inferenceStartTimeout: parseInt(process.env.INFERENCE_START_TIMEOUT || '30000', 10),
};

let inferenceProcess = null;
let inferenceProcessStartedByNode = false;

/**
 * Start Python Inference Process
 */
async function startInferenceProcess() {
    if (inferenceProcess) {
        console.log('Inference process already running');
        return;
    }

    console.log(`📣 Starting inference service: ${CONFIG.inferenceCmd} ${CONFIG.inferenceArgs.join(' ')}`);

    inferenceProcess = spawn(CONFIG.inferenceCmd, CONFIG.inferenceArgs, {
        cwd: __dirname,
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe']
    });
    inferenceProcessStartedByNode = true;

    inferenceProcess.stdout.on('data', (data) => {
        process.stdout.write(`[inference] ${data}`);
    });
    inferenceProcess.stderr.on('data', (data) => {
        process.stderr.write(`[inference] ${data}`);
    });

    inferenceProcess.on('exit', (code, signal) => {
        console.warn(`⚠️ Inference process exited: code=${code}, signal=${signal}`);
        inferenceProcess = null;
    });

    const startTime = Date.now();
    const timeout = CONFIG.inferenceStartTimeout;

    while (Date.now() - startTime < timeout) {
        const ok = await checkInferenceServer();
        if (ok) {
            console.log('✅ Inference service is ready');
            return;
        }
        await new Promise(r => setTimeout(r, 1000));
    }

    throw new Error(`Starting inference service timed out (${timeout} ms)`);
}

/**
 * Stop Inference Process
 */
function stopInferenceProcess() {
    if (!inferenceProcess) return;
    try {
        console.log('🛑 Stopping inference process...');
        inferenceProcess.kill('SIGTERM');
        setTimeout(() => {
            if (inferenceProcess) {
                try { inferenceProcess.kill('SIGKILL'); } catch (e) {}
                inferenceProcess = null;
            }
        }, 3000);
    } catch (err) {
        console.error('Failed to stop inference process:', err);
    }
}

/**
 * Check Inference Server Status
 */
async function checkInferenceServer() {
    try {
        const response = await axios.get(`${CONFIG.inferenceServerUrl}/health`, {
            timeout: 5000
        });
        return response.status === 200;
    } catch (error) {
        return false;
    }
}

/**
 * Call FastAPI Inference Service for Batch Prediction
 */
async function callInferencePredictBatch(texts) {
    try {
        const response = await axios.post(
            `${CONFIG.inferenceServerUrl}/predict-batch`,
            { texts: texts },
            { timeout: CONFIG.inferenceServerTimeout }
        );
        return response.data;
    } catch (error) {
        throw new Error(`Batch prediction error: ${error.response?.data?.detail || error.message}`);
    }
}

// ===== API Endpoints =====

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'Index.html'));
});

/**
 * GET /api/health - Health Check
 */
app.get('/api/health', async (req, res) => {
    try {
        const inferenceServerAvailable = await checkInferenceServer();
        
        res.json({
            status: inferenceServerAvailable ? 'ok' : 'degraded',
            timestamp: new Date().toISOString(),
            services: {
                nodeServer: 'running',
                inferenceServer: inferenceServerAvailable ? 'running' : 'unavailable'
            },
            features: {
                rag: true,
                batch: true
            },
            config: {
                inferenceServerUrl: CONFIG.inferenceServerUrl
            }
        });
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: error.message
        });
    }
});

/**
 * POST /api/predict - RAG Enhanced Prediction (Single Prediction Endpoint)
 * Body: { 
 *   "text": "your text here",
 *   "use_retrieval": true,  // default true
 *   "sources": ["wikipedia", "web"]  // optional
 * }
 */
app.post('/api/predict', async (req, res) => {
    try {
        const { text, use_retrieval = true, sources = null } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({
                error: 'Invalid input',
                message: 'Text must be a non-empty string'
            });
        }

        if (text.trim().length === 0) {
            return res.status(400).json({
                error: 'Invalid input',
                message: 'Text cannot be empty'
            });
        }

        console.log(`[RAG Prediction] ${text.substring(0, 50)}... (Retrieval: ${use_retrieval})`);
        
        const endpoint = `${CONFIG.inferenceServerUrl}/predict`;
        const requestBody = {
            text: text,
            use_retrieval: use_retrieval,
            sources: sources
        };

        const response = await axios.post(endpoint, requestBody, {
            timeout: CONFIG.inferenceServerTimeout
        });

        if (!response.data) {
            throw new Error('Empty response from inference server');
        }

        res.json({
            success: true,
            ...response.data
        });

    } catch (error) {
        console.error('RAG Prediction Error:', error.message);
        res.status(503).json({
            error: 'RAG prediction failed',
            message: error.response?.data?.detail || error.message
        });
    }
});

/**
 * POST /api/predict-batch - Batch Prediction
 * Body: { "texts": ["text1", "text2", ...] }
 */
app.post('/api/predict-batch', async (req, res) => {
    try {
        const { texts } = req.body;

        if (!Array.isArray(texts)) {
            return res.status(400).json({
                error: 'Invalid input',
                message: 'texts must be an array'
            });
        }

        if (texts.length === 0) {
            return res.status(400).json({
                error: 'Invalid input',
                message: 'texts array cannot be empty'
            });
        }

        if (texts.length > 100) {
            return res.status(400).json({
                error: 'Invalid input',
                message: 'Maximum 100 texts per request'
            });
        }

        for (let i = 0; i < texts.length; i++) {
            if (typeof texts[i] !== 'string' || texts[i].trim().length === 0) {
                return res.status(400).json({
                    error: 'Invalid input',
                    message: `Text at index ${i} is invalid`
                });
            }
        }

        console.log(`[Batch Prediction] ${texts.length} texts`);
        const result = await callInferencePredictBatch(texts);

        res.json({
            success: true,
            ...result
        });

    } catch (error) {
        console.error('Batch prediction error:', error.message);
        res.status(503).json({
            error: 'Batch prediction failed',
            message: error.message
        });
    }
});

// Error Handling
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: err.message
    });
});

// ===== Start Server =====
const PORT = process.env.PORT || 3000;

async function startServer() {
    console.log('='.repeat(80));
    console.log('🚀 Node.js Server Starting (RAG Enhanced)');
    console.log('='.repeat(80));
    
    console.log(`\n🔍 Checking inference service: ${CONFIG.inferenceServerUrl}`);
    let inferenceServerAvailable = await checkInferenceServer();

    if (inferenceServerAvailable) {
        console.log('✅ Inference service connected');
    } else {
        console.warn('⚠️ Inference service currently unavailable');

        if (CONFIG.autoStartInference) {
            console.log('⚙️ Automatically starting inference service...');
            try {
                await startInferenceProcess();
                inferenceServerAvailable = await checkInferenceServer();
                if (inferenceServerAvailable) console.log('✅ Inference service is ready');
            } catch (err) {
                console.error('Automatic start failed:', err.message);
                console.warn('Please start manually: python Inference.py');
            }
        } else {
            console.warn('Please start the FastAPI service manually');
        }
    }
    
    app.listen(PORT, () => {
        console.log('\n' + '='.repeat(80));
        console.log('✅ Server started successfully');
        console.log('='.repeat(80));
        console.log(`📡 Node.js API: http://localhost:${PORT}`);
        console.log(`🔗 Inference Service: ${CONFIG.inferenceServerUrl}`);
        console.log(`📊 Demo Page: http://localhost:${PORT}/Index.html`);
        console.log(`🧠 RAG Feature: Enabled`);
        console.log('='.repeat(80) + '\n');
    });
}

startServer().catch(err => {
    console.error('Startup failed:', err);
    process.exit(1);
});

process.on('exit', stopInferenceProcess);
process.on('SIGINT', () => {
    stopInferenceProcess();
    process.exit(0);
});
process.on('SIGTERM', () => {
    stopInferenceProcess();
    process.exit(0);
});
process.on('uncaughtException', (err) => {
    console.error('Uncaught exception:', err);
    stopInferenceProcess();
    process.exit(1);
});