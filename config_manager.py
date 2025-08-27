import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """
    Configuration for different model variants.
    """
    name: str
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512
    retrieval_k: int =4
    
@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation settings.
    """
    results_dir: str = "evaluation_results"
    save_response: bool = True
    include_metadata: bool = True
    timeout_seconds: int = 30
    
class ConfigManager:
    """
    Centralized configuration management for RAG system variants.
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.configs = {}
        self.load_configs()
        
    def load_configs(self):
        """Load configurations from file or create defaults.
        """
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.configs = data
        else:
            self.create_default_configs()
            self.save_configs()
            
    def create_default_configs(self):
        """
        Create default model configurations for A/B testing.
        """
        self.configs = {
            "models": {
                "baseline": {
                    "name": "baseling",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 512,
                    "retrieval_k": 4
                },
                "creative": {
                    "name": "creative",
                    "model": "gpt-4",
                    "temperature": 0.9,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 512,
                    "retrieval_k": 4
                },
                "conservative": {
                    "name": "conservative",
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 512,
                    "retrieval_k": 6
                },
                "fast": {
                    "name": "fast",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 512,
                    "retrieval_k": 3
                }
            },
            "evaluation": {
                "results_dir": "evaluation_results",
                "save_response": True,
                "include_metadata": True,
                "timeout_seconds": 30
            },
            "environment": {
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "pinecone_api_key": os.environ.get("PINECONE_API_KEY"),
                "index_name": os.environ.get("INDEX_NAME")
            }
        }
        
    def save_configs(self):
        """
        """