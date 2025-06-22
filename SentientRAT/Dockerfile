
FROM python:3.10-slim

# Install Node.js for React build
RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build React GUI
COPY gui/ ./gui/
RUN cd gui && npm install && npm run build

COPY . .

ENV DB_PATH=/app/memory/sentientrat.db
ENV CHROMADB_PATH=/app/memory/chromadb
ENV TOOL_CONFIG_PATH=/app/tools_config.yaml
ENV KNOWLEDGE_DIR=/app/knowledge
ENV LLM_MODEL=minimax/m1
ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main_agent:app", "--host", "0.0.0.0", "--port", "8000"]