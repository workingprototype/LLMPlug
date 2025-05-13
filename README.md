# LLMPlug

LLMPlug is a Node.js library designed to simplify the integration of various Large Language Models (LLMs) from different vendors into your applications. It provides a unified interface for common LLM tasks like text generation and chat completions.

## Features

-   **Unified API**: Use the same methods (`generate`, `chat`) for different LLM providers.
-   **Easy Configuration**: Simple setup for API keys and model preferences.
-   **Provider Support**:
    -   OpenAI (GPT models)
    -   Anthropic (Claude models)
    -   Google (Gemini models)
    -   Hugging Face (Inference API for various models)
-   **Extensible**: Designed to easily add new providers.
-   **Error Handling**: Custom error types for better debugging.

## Installation

```bash
npm install llmplug