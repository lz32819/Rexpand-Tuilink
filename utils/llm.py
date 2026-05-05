import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.base import LanguageModelInput
from rexpand_pyutils_file import read_file, write_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

default_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY)


def _resolve_cache_dir() -> Path:
    """
    Use a writable cache directory in deployed environments.
    """
    configured = os.getenv("LLM_CACHE_DIR")
    candidates = [Path(configured)] if configured else [Path("./.cache"), Path(tempfile.gettempdir()) / "tuilink-cache"]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("", encoding="utf-8")
            test_file.unlink()
            return candidate
        except OSError:
            continue

    raise OSError("No writable cache directory available")


CACHE_DIR = _resolve_cache_dir()


def invoke_llm(
    input: LanguageModelInput,
    config: Optional[RunnableConfig] = None,
    *,
    use_cache: bool = False,
    verbose: bool = False,
    llm: Optional[ChatOpenAI] = default_llm,
    **kwargs: Any,
) -> BaseMessage:
    if use_cache:
        # Create a hash of the input string
        input_hash = hashlib.md5((str(input) + "|" + str(config)).encode()).hexdigest()
        filepath = str(CACHE_DIR / f"{input_hash}.json")

        cached_response = read_file(filepath)
        if cached_response is not None:
            if verbose:
                logging.info(f"Cache hit: {filepath}")

            return AIMessage(**cached_response)
        else:
            if verbose:
                logging.info(f"Cache miss: {filepath}")

            response: BaseMessage = llm.invoke(input, config, **kwargs)
            write_file(filepath, response.model_dump())
            return response
    else:
        return llm.invoke(input, config, **kwargs)
