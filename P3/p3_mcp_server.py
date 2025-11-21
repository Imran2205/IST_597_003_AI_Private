"""
P3 RAG-Anything MCP Server
Combines terminal operations (P1) with RAG-Anything for knowledge management
Supports multiple file formats: PDF, DOCX, PNG, JPG, TXT, MD
Supports multiple parsers: MinerU and Docling
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
import json
import shutil

from mcp.server.fastmcp import FastMCP
import ollama
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("rag-anything-server", port=3000)

# Global RAG instance
class RAGManager:
    """
    Manages RAG-Anything instance for the server.
    Configuration is received from the client during initialization.
    """
    
    def __init__(self):
        self.rag = None
        self.initialized = False
        
        # These will be set during initialize() from client's env_vars
        self.openrouter_api_key = None
        self.ollama_api_key = None
        self.working_dir = None
        self.data_dir = None
        self.output_dir = None
        self.llm_model = None
        self.embedding_model = None
        self.vlm_model = None
        
        # Ollama client for embeddings (reuse connection)
        self.ollama_client = None
        
    async def ollama_llm(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """
        LLM function using Ollama
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        # Run synchronous ollama.chat in thread pool
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.llm_model,
            messages=messages
        )
        return response['message']['content']
    
    async def ollama_embedding(self, texts):
        """
        Embedding function using Ollama AsyncClient
        Ollama handles batching internally, so we just pass all texts at once
        """
        try:
            # Create async client
            async_client = ollama.AsyncClient()
            
            # Pass all texts at once - Ollama handles batching internally
            response = await async_client.embed(
                model=self.embedding_model,
                input=texts  # Can be a single string or list of strings
            )
            
            # Return the embeddings
            return response['embeddings']
            
        except Exception as e:
            logger.error(f"Error embedding {len(texts)} texts: {e}")
            # Return zero vectors on error to avoid complete failure
            return [[0.0] * 768 for _ in texts]
    
    async def openrouter_vision(self, prompt, system_prompt=None, history_messages=[], 
                          image_data=None, messages=None, **kwargs):
        """
        Vision model using OpenRouter's Polaris-Alpha (free VLM)
        """
        import aiohttp
        
        # Build messages for OpenRouter API
        if messages:
            # VLM-enhanced query format from RAG-Anything
            request_messages = [msg for msg in messages if msg is not None]
        elif image_data:
            # Single image format
            request_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }]
        else:
            # Text-only fallback to Ollama LLM
            return await self.ollama_llm(prompt, system_prompt, history_messages, **kwargs)
        
        # Call OpenRouter API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "RAG-Anything Knowledge Navigator"
                    },
                    json={
                        "model": self.vlm_model,
                        "messages": request_messages
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            # Fallback to text-only response
            return f"[Vision analysis unavailable: {str(e)}]\n\n{await self.ollama_llm(prompt, system_prompt, history_messages, **kwargs)}"
    
    async def initialize(
        self, 
        env_vars: dict,
        llm_model: str,
        embedding_model: str,
        vlm_model: str,
        enable_vision: bool = True,
        parser: str = "mineru"
    ):
        """
        Initialize RAG-Anything instance with configuration from client.
        
        Args:
            env_vars: Dictionary containing environment variables from client
            llm_model: LLM model name to use
            embedding_model: Embedding model name to use
            vlm_model: Vision-language model name to use
            enable_vision: Whether to enable vision model support
            parser: Parser to use - "mineru" or "docling"
        """
        if self.initialized:
            return "RAG system already initialized"
        
        # Extract configuration from env_vars
        self.openrouter_api_key = env_vars.get("OPENROUTER_API_KEY")
        self.ollama_api_key = env_vars.get("OLLAMA_API_KEY")
        self.working_dir = env_vars.get("RAG_STORAGE_DIR")
        self.data_dir = env_vars.get("DATA_DIR")
        self.output_dir = env_vars.get("OUTPUT_DIR")
        
        # Set model names
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vlm_model = vlm_model
        
        # Validate required configuration
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.ollama_api_key:
            missing.append("OLLAMA_API_KEY")
        if not self.working_dir:
            missing.append("RAG_STORAGE_DIR")
        if not self.data_dir:
            missing.append("DATA_DIR")
        if not self.output_dir:
            missing.append("OUTPUT_DIR")
        if not self.llm_model:
            missing.append("llm_model")
        if not self.embedding_model:
            missing.append("embedding_model")
        if not self.vlm_model:
            missing.append("vlm_model")
        
        if missing:
            error_msg = f"Missing required configuration: {', '.join(missing)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        
        logger.info(f"Initializing RAG-Anything system with parser: {parser}...")
        logger.info(f"Configuration received from client:")
        logger.info(f"  LLM: {self.llm_model}")
        logger.info(f"  Embedding: {self.embedding_model}")
        logger.info(f"  VLM: {self.vlm_model}")
        logger.info(f"  Data Dir: {self.data_dir}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Storage Dir: {self.working_dir}")
        
        # Check if OpenRouter API key is available for vision
        if enable_vision and not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not found. Disabling vision support.")
            enable_vision = False
        
        # Create directories
        Path(self.working_dir).mkdir(exist_ok=True, parents=True)
        Path(self.data_dir).mkdir(exist_ok=True, parents=True)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        # Check for existing data BEFORE initialization
        storage_path = Path(self.working_dir)
        existing_data = False
        if storage_path.exists():
            vdb_files = list(storage_path.glob("vdb_*.json"))
            kv_files = list(storage_path.glob("kv_store_*.json"))
            existing_data = bool(vdb_files or kv_files)
        
        if existing_data:
            logger.info(f"📦 Found existing knowledge graph in {self.working_dir}")
            logger.info("Will automatically load stored data...")
        else:
            logger.info(f"📦 No existing data found - starting fresh knowledge base")
        
        # Configure RAG-Anything
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser=parser,  # "mineru" or "docling"
            parse_method="auto",
            enable_image_processing=enable_vision,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        # Set up embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=self.ollama_embedding
        )
        
        # Initialize RAG-Anything
        if enable_vision:
            self.rag = RAGAnything(
                config=config,
                llm_model_func=self.ollama_llm,
                embedding_func=embedding_func,
                vision_model_func=self.openrouter_vision,
            )
            vision_info = f"Vision: {self.vlm_model} (via OpenRouter)"
        else:
            self.rag = RAGAnything(
                config=config,
                llm_model_func=self.ollama_llm,
                embedding_func=embedding_func,
            )
            vision_info = "Vision: Disabled"
        
        self.initialized = True
        models_info = f"LLM: {self.llm_model}\nEmbedding: {self.embedding_model}\n{vision_info}\nParser: {parser}"
        logger.info(f"RAG-Anything initialized with:\n{models_info}")
        
        # Build status message
        status_msg = (
            f"✅ RAG system initialized successfully\n\n"
            f"{models_info}"
        )
        
        if existing_data:
            status_msg += (
                f"\n\n📚 Loaded existing knowledge graph from storage"
                f"\nStorage: {self.working_dir}"
                f"\n\n✅ Ready to query immediately - no need to reprocess documents!"
            )
        else:
            status_msg += (
                f"\n\n⚠️ No existing data - use preprocess_documents to build knowledge base"
            )
        
        return status_msg

# Global RAG manager instance
rag_manager = RAGManager()

#region Terminal Tools (from P1)
import subprocess

user_home = os.getenv('HOME')
proc = None

@mcp.tool()
async def initiate_terminal(cwd: str = ""):
    """
    Initiate a new terminal. Use this tool when running a command for the first time.
    By default initiates a bash terminal in the data directory.
    
    Args:
        cwd: directory where the terminal is initiated
    """
    global proc
    if proc is not None:
        proc.terminate()
        proc.wait()

    if cwd != "":
        if "~" in cwd:
            cwd = cwd.replace("~", user_home)
        if not os.path.isdir(cwd):
            return f"{cwd} is not a directory"
        proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    else:
        # Default to data directory
        data_dir = os.getenv("DATA_DIR", user_home)
        proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=data_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    return f"Terminal initiated in: {cwd if cwd else data_dir}"

@mcp.tool()
async def terminate_terminal():
    """
    Terminates the opened terminal
    """
    global proc
    if proc is not None:
        proc.terminate()
        proc.wait()
        proc = None
        return "Terminal terminated"
    return "No terminal is open"

@mcp.tool()
async def run_command(command: str) -> str:
    """
    Runs a command in the terminal and returns the output as a string.
    
    Args:
        command: bash command to run in terminal
    """
    global proc

    try:
        output = []
        if proc is not None:
            if not command.endswith("\n"):
                command += "\n"

            proc.stdin.write(command)

            marker = "[END_OF_CMD]"
            proc.stdin.write(f"echo {marker}\n")
                    
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line: 
                    break
                if marker in line: 
                    break
                output.append(line.rstrip())        
        else:
            output = ["No terminal has been initiated. Please initiate a terminal first with `initiate_terminal(working_dir)`"]

        out = "\n".join(output)
        output_str = (
            f"Command: {command}\n"
            f"Output: {out}"
        )

        return output_str
    
    except Exception as e:
        return f"ERROR: An exception occurred {type(e).__name__}. Details: {e}"

#endregion Terminal Tools

#region RAG-Anything Tools

def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    env_vars = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars

@mcp.tool()
async def initialize_rag_system(env_vars: dict, llm_model: str, embedding_model: str, vlm_model: str, enable_vision: bool = True, parser: str = "mineru") -> str:
    """
    Initialize the RAG-Anything system with configuration from client.
    Must be called before any other RAG operations.
    
    Args:
        env_vars: Dictionary containing environment variables (API keys, directories)
        llm_model: LLM model name to use (e.g., "gpt-oss:20b-cloud")
        embedding_model: Embedding model name to use (e.g., "nomic-embed-text")
        vlm_model: Vision-language model name to use (e.g., "openrouter/polaris-alpha")
        enable_vision: Whether to enable vision model support
        parser: Parser to use - "mineru" (default) or "docling"
    
    Returns:
        Status message
    """
    # print(">>>>", env_vars)
    if 'OPENROUTER_API_KEY' not in list(env_vars.keys()):
        env_vars = load_env_file()
        vlm_model = 'openrouter/polaris-alpha'
        llm_model = "gpt-oss:20b-cloud"
        embedding_model = "nomic-embed-text"
        # print("+++", env_vars)
    try:
        # Validate parser choice
        if parser not in ["mineru", "docling"]:
            return f"Error: Invalid parser '{parser}'. Choose 'mineru' or 'docling'"
        
        result = await rag_manager.initialize(
            env_vars=env_vars,
            llm_model=llm_model,
            embedding_model=embedding_model,
            vlm_model=vlm_model,
            enable_vision=enable_vision,
            parser=parser
        )
        return result
    except Exception as e:
        error_msg = f"Error initializing RAG system: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def preprocess_documents( data_directory: str = None,file_extensions: list = None) -> str:
    """
    Preprocess all documents in the specified directory using RAG-Anything.
    Supports: PDF, DOCX, PNG, JPG, JPEG, TXT, MD
    
    This tool will:
    1. Find all supported files in the directory
    2. Parse them using the configured parser (MinerU or Docling)
    3. Process with RAG-Anything
    4. Store them in persistent vector database
    
    Args:
        data_directory: Path to directory containing files (default: from .env DATA_DIR)
        file_extensions: List of file extensions to process (default: [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md"])
    
    Returns:
        Status message with processing results
    """
    try:
        if not rag_manager.initialized:
            return "Error: RAG system not initialized. Please call initialize_rag_system first."
        
        # Use environment variable if not specified
        if data_directory is None:
            data_directory = rag_manager.data_dir
        
        data_path = Path(data_directory)
        
        if not data_path.exists():
            return f"Error: Directory {data_directory} does not exist"
        
        # Default file extensions if not specified
        if file_extensions is None:
            file_extensions = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md"]
        
        # Find all supported files
        all_files = []
        for ext in file_extensions:
            all_files.extend(list(data_path.rglob(f"*{ext}")))
        
        if not all_files:
            return (
                f"No supported files found in {data_directory}\n"
                f"Supported: {', '.join(file_extensions)}"
            )
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process each file
        processed = 0
        failed = 0
        results = []
        
        for file_path in all_files:
            try:
                logger.info(f"Processing: {file_path.name}")
                
                # Process with RAG-Anything
                await rag_manager.rag.process_document_complete(
                    file_path=str(file_path),
                    output_dir=rag_manager.output_dir,
                    parse_method="auto"
                )
                
                processed += 1
                results.append(f"✅ {file_path.name}")
                
            except Exception as e:
                failed += 1
                results.append(f"❌ {file_path.name}: {str(e)}")
                logger.error(f"Error processing {file_path.name}: {e}")
        
        summary = (
            f"Preprocessing Complete:\n"
            f"- Total files: {len(all_files)}\n"
            f"- Successfully processed: {processed}\n"
            f"- Failed: {failed}\n"
            f"- Knowledge base stored in: {rag_manager.working_dir}\n\n"
            f"Results:\n"
            f"{chr(10).join(results[:20])}\n"
            f"{'...' if len(results) > 20 else ''}\n\n"
            f"You can now query the knowledge base using query_knowledge tool."
        )
        
        return summary.strip()
        
    except Exception as e:
        error_msg = f"Error preprocessing documents: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def query_knowledge(question: str, mode: str = "hybrid") -> str:
    """
    Query the RAG-Anything knowledge base for information from processed documents.
    
    Args:
        question: Question to ask about the documents
        mode: Query mode - "hybrid" (default), "local", "global", or "naive"
    
    Returns:
        Answer based on the knowledge base
    """
    try:
        if not rag_manager.initialized:
            return "Error: RAG system not initialized. Please call initialize_rag_system first."
        
        if rag_manager.rag is None:
            return "Error: No documents have been processed yet. Please run preprocess_documents first."
        
        logger.info(f"Querying knowledge base: {question}")
        
        # Query RAG-Anything
        result = await rag_manager.rag.aquery(question, mode=mode)
        
        return result
        
    except Exception as e:
        error_msg = f"Error querying knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def add_document(file_path: str) -> str:
    """
    Add a new document to the existing knowledge base.
    Supports: PDF, DOCX, PNG, JPG, JPEG, TXT, MD
    
    Args:
        file_path: Path to the file to add
    
    Returns:
        Status message
    """
    try:
        if not rag_manager.initialized:
            return "Error: RAG system not initialized. Please call initialize_rag_system first."
        
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        # Check if file extension is supported
        supported_extensions = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md"]
        if path.suffix.lower() not in supported_extensions:
            return (
                f"Error: Unsupported file type {path.suffix}. "
                f"Supported: {', '.join(supported_extensions)}"
            )
        
        logger.info(f"Adding document: {path.name}")
        
        # Process with RAG-Anything
        await rag_manager.rag.process_document_complete(
            file_path=str(path),
            output_dir=rag_manager.output_dir,
            parse_method="auto"
        )
        
        return f"✅ Successfully added {path.name} to the knowledge base"
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def get_rag_status(detailed: bool = False) -> str:
    """
    Get comprehensive status of the RAG-Anything system including:
    - System initialization status
    - Directory configuration
    - Files in data directory
    - Processed documents in storage
    - Knowledge graph statistics
    
    Args:
        detailed: If True, shows list of processed files (default: False)
    
    Returns:
        Complete system status information
    """
    try:
        # === System Configuration ===
        status = "🤖 P3 RAG-Anything System Status\n\n"
        
        status += "📋 Configuration:\n"
        status += f"  • Server: ✅ Running\n"
        status += f"  • RAG Initialized: {'✅ Yes' if rag_manager.initialized else '❌ No'}\n"
        status += f"  • Data Directory: {rag_manager.data_dir or 'Not set'}\n"
        status += f"  • Output Directory: {rag_manager.output_dir or 'Not set'}\n"
        status += f"  • Storage Directory: {rag_manager.working_dir or 'Not set'}\n"
        
        if rag_manager.initialized:
            status += f"  • LLM Model: {rag_manager.llm_model}\n"
            status += f"  • Embedding Model: {rag_manager.embedding_model}\n"
            status += f"  • VLM Model: {rag_manager.vlm_model}\n"
        
        # === Files in Data Directory ===
        status += "\n📁 Files in Data Directory:\n"
        data_path = Path(rag_manager.data_dir) if rag_manager.data_dir else None
        supported_extensions = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md"]
        file_counts = {}
        total_files = 0
        
        if data_path and data_path.exists():
            for ext in supported_extensions:
                count = len(list(data_path.rglob(f"*{ext}")))
                if count > 0:
                    file_counts[ext] = count
                    total_files += count
            
            if file_counts:
                for ext, count in file_counts.items():
                    status += f"  • {ext}: {count}\n"
                status += f"  Total: {total_files} files\n"
            else:
                status += "  • No supported files found\n"
        else:
            status += "  • Data directory not accessible\n"
        
        # === Processed Documents in Storage ===
        status += "\n💾 Processed Documents:\n"
        
        if rag_manager.working_dir:
            storage_path = Path(rag_manager.working_dir)
            full_docs_file = storage_path / "kv_store_full_docs.json"
            
            if full_docs_file.exists():
                try:
                    with open(full_docs_file, 'r') as f:
                        full_docs = json.load(f)
                    
                    print(full_docs)
                    doc_count = len(full_docs)
                    status += f"  • Total Processed: {doc_count}\n"
                    
                    # Count by status
                    status_counts = {}
                    processed_files = []
                    
                    for doc_id, doc_data in full_docs.items():
                        file_path = doc_data.get('file_path', 'Unknown')
                        doc_status = doc_data.get('status', 'unknown')                        
                        chunks_count = doc_data.get('chunks_count', 0)
                        processed_files.append({
                            'file': Path(file_path).name if file_path != 'Unknown' else 'Unknown'
                            # 'status': doc_status,
                            # 'chunks': chunks_count
                        })
                
                    
                    
                    # Show file list if detailed
                    if processed_files:
                        status += "\n  📄 Processed Files:\n"
                        for pf in processed_files[:20]:  # Show up to 20
                            # status += f"    • {pf['file']} ({pf['status']}, {pf['chunks']} chunks)\n"
                            status += f"    • {pf['file']} \n"
                        if len(processed_files) > 20:
                            status += f"    ... and {len(processed_files) - 20} more\n"
                    
                except Exception as e:
                    status += f"  • Error reading storage: {e}\n"
            else:
                status += "  • No documents processed yet\n"
        else:
            status += "  • Storage not configured\n"
        
        print(status)
        return status.strip()
        
    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def clear_rag_storage() -> str:
    """
    Clear all RAG storage and start fresh. WARNING: This deletes all indexed data!
    
    Returns:
        Status message
    """
    try:
        if not rag_manager.working_dir:
            return "Error: RAG system not configured"
        
        storage_path = Path(rag_manager.working_dir)
        
        if not storage_path.exists():
            return "Storage directory doesn't exist"
        
        # Count files before deletion
        files = list(storage_path.glob("*"))
        count = len(files)
        
        # Delete all files
        for file in files:
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {file}: {e}")
        
        return (
            f"✅ Cleared {count} files from storage. "
            f"Knowledge base is now empty. Run preprocess_documents to rebuild."
        )
        
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def rename_pdfs_by_title(directory: str = None) -> str:
    """
    Rename PDF files in the directory based on their document titles.
    Uses PyPDF2 to extract title metadata and sanitizes filenames.
    
    Args:
        directory: Directory containing PDF files to rename (default: from .env DATA_DIR)
    
    Returns:
        Summary of renamed files
    """
    try:
        from PyPDF2 import PdfReader
        import re
        
        # Use environment variable if not specified
        if directory is None:
            directory = rag_manager.data_dir
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"Error: Directory {directory} does not exist"
        
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            return f"No PDF files found in {directory}"
        
        renamed = []
        skipped = []
        
        def sanitize_filename(title):
            # Remove invalid characters
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            # Limit length
            if len(title) > 200:
                title = title[:200]
            return title.strip()
        
        for pdf_file in pdf_files:
            try:
                reader = PdfReader(pdf_file)
                
                # Try to get title from metadata
                title = None
                if reader.metadata and reader.metadata.title:
                    title = reader.metadata.title
                else:
                    # Try first page as fallback
                    if len(reader.pages) > 0:
                        first_page_text = reader.pages[0].extract_text()
                        lines = first_page_text.split('\n')
                        for line in lines[:5]:
                            line = line.strip()
                            if 10 < len(line) < 200:
                                title = line
                                break
                
                if not title:
                    skipped.append(f"⏭️  {pdf_file.name} (no title found)")
                    continue
                
                # Sanitize and create new filename
                new_name = sanitize_filename(title) + ".pdf"
                new_path = pdf_file.parent / new_name
                
                # Avoid overwriting
                if new_path.exists() and new_path != pdf_file:
                    counter = 1
                    while new_path.exists():
                        new_name = f"{sanitize_filename(title)}_{counter}.pdf"
                        new_path = pdf_file.parent / new_name
                        counter += 1
                
                if new_path == pdf_file:
                    skipped.append(f"⏭️  {pdf_file.name} (already named)")
                else:
                    pdf_file.rename(new_path)
                    renamed.append(f"✅ {pdf_file.name} → {new_name}")
                
            except Exception as e:
                skipped.append(f"❌ {pdf_file.name}: {str(e)}")
        
        summary = (
            f"PDF Renaming Complete:\n"
            f"- Total files: {len(pdf_files)}\n"
            f"- Renamed: {len(renamed)}\n"
            f"- Skipped: {len(skipped)}\n\n"
            f"Renamed:\n"
            f"{chr(10).join(renamed[:15]) if renamed else 'None'}\n"
            f"{'...' if len(renamed) > 15 else ''}\n\n"
            f"Skipped:\n"
            f"{chr(10).join(skipped[:15]) if skipped else 'None'}\n"
            f"{'...' if len(skipped) > 15 else ''}"
        )
        
        return summary.strip()
        
    except Exception as e:
        error_msg = f"Error renaming PDFs: {str(e)}"
        logger.error(error_msg)
        return error_msg

#endregion RAG-Anything Tools

# Main
if __name__ == "__main__":
    logger.info("Starting P3 RAG-Anything MCP Server...")
    logger.info("Configuration will be received from client during initialization")
    logger.info("Combining terminal operations with RAG-Anything for knowledge management")
    logger.info("Supported file types: PDF, DOCX, PNG, JPG, JPEG, TXT, MD")
    logger.info("Supported parsers: MinerU, Docling")
    logger.info("Transport: streamable-http on port 3000")
    
    # Run the server with streamable-http transport
    mcp.run(transport="streamable-http")