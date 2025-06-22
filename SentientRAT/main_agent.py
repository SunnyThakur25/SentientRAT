import subprocess
import sqlite3
import yaml
import requests
import logging
import os
import re
import glob
import psutil
import uvicorn
import platform
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader, UnstructuredLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import VLLM, HuggingFacePipeline
from langchain.tools import Tool
from shutil import which
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(filename="logs/sentientrat.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI
app = FastAPI(title="SentientRAT API")

# OAuth2 for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Environment variables
DB_PATH = os.getenv("DB_PATH", "memory/sentientrat.db")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "memory/chromadb")
TOOL_CONFIG_PATH = os.getenv("TOOL_CONFIG_PATH", "tools_config.yaml")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")
LLM_MODEL = os.getenv("LLM_MODEL", "minimax/m1")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Initialize LLM
def init_llm(model_name):
    try:
        if model_name in ["minimax/m1", "deepseek/r1"]:
            return VLLM(model=model_name, max_context_length=1000000, temperature=0.7)
        elif model_name == "meta-llama/llama-3.1-8b":
            return HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation", pipeline_kwargs={"max_length": 4096})
        else:
            logging.error(f"Unsupported LLM: {model_name}")
            return VLLM(model="minimax/m1", max_context_length=1000000, temperature=0.7)
    except Exception as e:
        logging.error(f"Failed to initialize LLM {model_name}: {e}")
        return None

llm = init_llm(LLM_MODEL)

# Initialize embeddings and memory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
docker_client = docker.from_env()

# Load tool config
try:
    with open(TOOL_CONFIG_PATH, "r") as f:
        tool_config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Failed to load tool config: {e}")
    tool_config = {}

# Initialize memory database
def init_memory_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations 
                     (id INTEGER PRIMARY KEY, timestamp TEXT, user_input TEXT, response TEXT)''')
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to initialize memory DB: {e}")
    finally:
        conn.close()

# Store/retrieve conversation
def store_conversation(user_input, response):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO conversations (timestamp, user_input, response) VALUES (?, ?, ?)", 
                  (datetime.now().isoformat(), user_input, response))
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to store conversation: {e}")
    finally:
        conn.close()

def retrieve_conversation(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT user_input, response FROM conversations WHERE user_input LIKE ?", (f"%{query}%",))
        results = c.fetchall()
        return results
    except Exception as e:
        logging.error(f"Failed to retrieve conversation: {e}")
        return []
    finally:
        conn.close()

# RAG for knowledge
def load_knowledge():
    try:
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        documents = []
        for file_path in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf")):
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
        for file_path in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt")):
            loader = UnstructuredLoader(file_path)
            documents.extend(loader.load())
        if documents:
            vector_store = Chroma.from_documents(documents, embeddings, persist_directory=CHROMADB_PATH)
            vector_store.persist()
            return vector_store
        return None
    except Exception as e:
        logging.error(f"Failed to load knowledge: {e}")
        return None

def web_search(query):
    try:
        searches = [
            requests.get(f"https://api.duckduckgo.com/?q={query}&format=json"),
            requests.get(f"https://api.search.brave.com/search?q={query}", headers={"X-Subscription-Token": os.getenv("BRAVE_API_KEY")})
        ]
        results = []
        for response in searches:
            if response.status_code == 200:
                if "duckduckgo" in response.url:
                    results.append(response.json().get("AbstractText", ""))
                elif "brave" in response.url:
                    results.append(response.json().get("web", {}).get("results", [{}])[0].get("description", ""))
        summary_prompt = f"Summarize: {' '.join([r for r in results if r])}"
        summary = llm(summary_prompt).strip()
        return {"raw": results, "summary": summary}
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return {"raw": [], "summary": "Web search failed"}

# Tool execution
def check_tool_installed(tool_name):
    return which(tool_name) is not None

def install_tool(tool_name):
    os_type = platform.system().lower()
    cmd = tool_config.get(tool_name, {}).get("install_cmd", {}).get(os_type, f"apt-get install {tool_name} -y")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        logging.info(f"Installed {tool_name}: {result.stdout}")
        return {"status": "success" if result.returncode == 0 else "error", "output": result.stdout, "error": result.stderr}
    except Exception as e:
        logging.error(f"Failed to install {tool_name}: {e}")
        return {"status": "error", "error": str(e)}

def run_tool(command):
    try:
        container = docker_client.containers.run("ubuntu:latest", command, detach=True, tty=True)
        logs = container.logs().decode("utf-8")
        container.remove()
        logging.info(f"Executed command {command}: {logs}")
        return {"status": "success", "output": logs}
    except Exception as e:
        logging.error(f"Failed to run tool: {e}")
        return {"status": "error", "error": str(e)}

# Canvas visualization
def generate_canvas(data, tool):
    try:
        parsed_data = {"x": [], "y": [], "labels": []}
        if tool in tool_config and "output_parser" in tool_config[tool]:
            parser = tool_config[tool]["output_parser"]
            for line in data.split("\n"):
                for key, pattern in parser.items():
                    match = re.search(pattern, line)
                    if match:
                        parsed_data[key].append(match.group(1))
        
        if not parsed_data["x"]:
            parsed_data["x"] = [i for i in range(1, min(len(data.split("\n")) + 1, 10))]
            parsed_data["y"] = [1] * len(parsed_data["x"])
            parsed_data["labels"] = data.split("\n")[:len(parsed_data["x"])]

        return f"""
        <div id="networkMap" style="width: 100%; height: 100vh;"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var data = [{{
                x: {parsed_data["x"]},
                y: {parsed_data["y"]},
                type: 'scatter',
                mode: 'markers+text',
                text: {parsed_data["labels"]},
                marker: {{ size: 12, color: 'green' }}
            }}];
            var layout = {{
                title: '{tool or "Tool"} Output Visualization',
                xaxis: {{ title: 'Index' }},
                yaxis: {{ title: 'Value' }},
                responsive: true,
                hovermode: 'closest'
            }};
            Plotly.newPlot('networkMap', data, layout);
        </script>
        """
    except Exception as e:
        logging.error(f"Failed to generate canvas: {e}")
        return ""

# Terminal emulator
def emulate_terminal(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        logging.info(f"Emulated terminal command {command}: {result.stdout}")
        return {"status": "success", "output": result.stdout, "error": result.stderr}
    except Exception as e:
        logging.error(f"Terminal emulation failed: {e}")
        return {"status": "error", "error": str(e)}

# LangChain tool definitions
tools = [
    Tool(
        name="run_tool",
        func=run_tool,
        description="Execute a pentesting tool or command in a sandboxed environment."
    ),
    Tool(
        name="web_search",
        func=web_search,
        description="Perform a web search using DuckDuckGo and Brave APIs."
    ),
    Tool(
        name="emulate_terminal",
        func=emulate_terminal,
        description="Run a command in a local terminal emulator."
    )
]

# Initialize conversational chain
def init_conversational_chain(vector_store):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2}) if vector_store else None
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        return chain
    except Exception as e:
        logging.error(f"Failed to initialize conversational chain: {e}")
        return None

# Dynamic command generation
def generate_command(user_input, entities, vector_store):
    try:
        tool = entities.get("tool")
        target = entities.get("target")
        flags = entities.get("flags", "")
        code = entities.get("code")
        file = entities.get("file")

        rag_result = rag_query(f"{tool} command options", vector_store) if tool else {"local": [], "web": ""}
        tool_info = rag_result["local"] + [rag_result["web"]]

        prompt = f"""
        User input: '{user_input}'
        Tool: {tool or 'unknown'}
        Target: {target or 'none'}
        Flags: {flags or 'none'}
        Code: {code or 'none'}
        File: {file or 'none'}
        Tool info: {tool_info}
        Generate a valid command for the requested action. Use tool_config.yaml if available.
        Return only the command string or None if invalid.
        """
        command = llm(prompt).strip()
        if command == "None" or not command:
            return None
        
        if tool and tool in tool_config:
            valid_cmd = tool_config[tool].get("command_template", "{tool} {flags} {target}")
            command = valid_cmd.format(tool=tool, flags=flags, target=target or "")
        
        return command
    except Exception as e:
        logging.error(f"Failed to generate command: {e}")
        return None

# Dynamic entity extraction
def parse_natural_language(user_input, vector_store):
    try:
        prompt = f"""
        Analyze the user input: '{user_input}'
        Identify the intent (e.g., scan, open, summarize, search, recall, execute).
        Extract entities: tool, target, flags, file, code, language.
        Return JSON: {{ "intent": str, "entities": {{ "tool": str, "target": str, "flags": str, "file": str, "code": str, "lang": str }} }}
        """
        response = llm(prompt)
        result = eval(response)
        
        entities = result["entities"]
        if not entities.get("tool"):
            tool_pattern = r"\b(\w+)\b"
            tool_match = re.search(tool_pattern, user_input.lower())
            if tool_match and tool_match.group(1) in tool_config:
                entities["tool"] = tool_match.group(1)
        if not entities.get("target"):
            ip_pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?|https?://[^\s]+)"
            ip_match = re.search(ip_pattern, user_input)
            if ip_match:
                entities["target"] = ip_match.group(1)
        if not entities.get("flags"):
            flag_pattern = r"(-[a-zA-Z]+|--[a-zA-Z-]+)"
            entities["flags"] = " ".join(re.findall(flag_pattern, user_input)) or ""
        if not entities.get("file"):
            file_pattern = r"(\w+\.\w+)"
            file_match = re.search(file_pattern, user_input)
            if file_match:
                entities["file"] = file_match.group(1)
        if not entities.get("lang") or not entities.get("code"):
            code_pattern = r"(python|bash)\s+code\s+(.+)"
            code_match = re.search(code_pattern, user_input, re.IGNORECASE)
            if code_match:
                entities["lang"] = code_match.group(1).lower()
                entities["code"] = code_match.group(2)
        
        return result
    except Exception as e:
        logging.error(f"Failed to parse natural language: {e}")
        return None

# Process command
def process_command(user_input, vector_store):
    try:
        memory_context = retrieve_conversation(user_input)
        context = memory_context[0][1] if memory_context else ""
        
        parsed = parse_natural_language(user_input, vector_store)
        if not parsed:
            response = {"status": "error", "message": "Could not understand your request"}
            store_conversation(user_input, str(response))
            return response

        intent = parsed["intent"]
        entities = parsed["entities"]
        response = {}

        chain = init_conversational_chain(vector_store)
        if not chain:
            response = {"status": "error", "message": "Failed to initialize conversational chain"}
            store_conversation(user_input, str(response))
            return response

        if intent in ["scan", "open"]:
            tool = entities.get("tool")
            if not tool:
                response = {"status": "error", "message": "No tool specified"}
            else:
                if not check_tool_installed(tool):
                    install_result = install_tool(tool)
                    if install_result["status"] == "error":
                        store_conversation(user_input, str(install_result))
                        return install_result
                command = generate_command(user_input, entities, vector_store)
                if not command:
                    response = {"status": "error", "message": "Invalid command generated"}
                else:
                    response = run_tool(command)
                    if intent == "scan":
                        response["canvas"] = generate_canvas(response.get("output", ""), tool)

        elif intent == "summarize":
            file_path = entities.get("file")
            if not file_path:
                response = chain({"question": "Summarize all pentesting documents in the knowledge directory"})
            else:
                vector_store = load_knowledge(os.path.join(KNOWLEDGE_DIR, file_path))
                if vector_store:
                    response = chain({"question": f"Summarize {file_path}"})
                else:
                    response = {"status": "error", "message": "Failed to load document"}

        elif intent == "search":
            query = user_input.replace("search", "").strip()
            if not query:
                response = {"status": "error", "message": "No search query provided"}
            else:
                response = chain({"question": query, "tool": "web_search"})

        elif intent == "recall":
            query = user_input.replace("recall", "").strip()
            if not query:
                response = {"status": "error", "message": "No recall query provided"}
            else:
                response = {"memory": retrieve_conversation(query)}

        elif intent == "execute":
            lang = entities.get("lang")
            code = entities.get("code")
            if not lang or not code:
                response = {"status": "error", "message": "Missing language or code"}
            else:
                command = generate_command(user_input, entities, vector_store)
                if not command:
                    response = {"status": "error", "message": "Invalid code command"}
                else:
                    response = emulate_terminal(command)

        response["context"] = context
        store_conversation(user_input, str(response))
        return response
    
    except Exception as e:
        logging.error(f"Command processing failed: {e}")
        response = {"status": "error", "message": str(e)}
        store_conversation(user_input, str(response))
        return response

# System info endpoint
@app.get("/system_info")
async def get_system_info():
    try:
        gpu_available = False
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            gpu_available = True
        except:
            pass

        return {
            "os": platform.system(),
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_available": gpu_available,
            "recommended_models": ["meta-llama/llama-3.1-8b"] if not gpu_available else ["minimax/m1", "deepseek/r1", "meta-llama/llama-3.1-8b"]
        }
    except Exception as e:
        logging.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model download endpoint
@app.post("/download_model")
async def download_model(model_name: str):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        cmd = f"huggingface-cli download {model_name} --cache-dir {MODEL_DIR}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Downloaded model {model_name}: {result.stdout}")
            return {"status": "success", "output": result.stdout}
        else:
            logging.error(f"Failed to download model {model_name}: {result.stderr}")
            return {"status": "error", "error": result.stderr}
    except Exception as e:
        logging.error(f"Model download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve GUI
@app.get("/")
async def serve_gui():
    return FileResponse("gui/index.html")

# Pydantic model for API input
class CommandInput(BaseModel):
    user_input: str

# Process command endpoint
@app.post("/process_command")
async def api_process_command(command: CommandInput, token: str = Depends(oauth2_scheme)):
    vector_store = load_knowledge()
    result = process_command(command.user_input, vector_store)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

if __name__ == "__main__":
    
    init_memory_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)