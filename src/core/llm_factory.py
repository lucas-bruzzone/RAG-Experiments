"""LLM Factory - Auto-detects and creates LLM instances"""

import os
import subprocess
from typing import Optional
from langchain_community.llms import Ollama, OpenAI


class LLMFactory:
    """Factory for creating LLM instances with auto-detection"""

    @staticmethod
    def _check_ollama() -> bool:
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    @staticmethod
    def _check_openai() -> bool:
        """Check if OpenAI API key is configured"""
        return bool(os.getenv("OPENAI_API_KEY"))

    @classmethod
    def create(cls, config: dict, provider: Optional[str] = None):
        """
        Create LLM instance based on configuration
        
        Args:
            config: LLM configuration dict
            provider: Force specific provider ('ollama', 'openai', or None for auto)
        """
        llm_config = config.get("llm", {})
        
        # Auto-detect if not specified
        if provider == "auto" or provider is None:
            if cls._check_ollama():
                provider = "ollama"
            elif cls._check_openai() and llm_config.get("openai", {}).get("use_if_available"):
                provider = "openai"
            else:
                raise RuntimeError(
                    "No LLM available. Install Ollama (https://ollama.ai) "
                    "or set OPENAI_API_KEY environment variable."
                )

        # Create LLM
        if provider == "ollama":
            ollama_config = llm_config.get("ollama", {})
            print(f"✓ Using Ollama: {ollama_config.get('model', 'tinyllama')}")
            return Ollama(
                model=ollama_config.get("model", "tinyllama"),
                base_url=ollama_config.get("base_url", "http://localhost:11434"),
                temperature=ollama_config.get("temperature", 0.7)
            )
        
        elif provider == "openai":
            openai_config = llm_config.get("openai", {})
            print(f"✓ Using OpenAI: {openai_config.get('model', 'gpt-4o-mini')}")
            return OpenAI(
                model=openai_config.get("model", "gpt-4o-mini"),
                temperature=openai_config.get("temperature", 0.7),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
