# LLMPlug

LLMPlug is a Node.js library designed to simplify the integration of various Large Language Models (LLMs) from different vendors and local setups into your applications. It provides a unified interface for common LLM tasks like text generation and chat completions, with advanced features.

## Features

-   **Unified API**: Use consistent methods (`generate`, `chat`, `generateStream`, `chatStream`) across diverse LLM providers.
-   **Easy Configuration**: Simple setup for API keys, local server URLs, and model preferences.
-   **Streaming Responses**: Get partial responses as they are generated.
-   **Function Calling (Tool Use)**: Enable LLMs to interact with external tools and APIs.
-   **Multimodal Input**: Send images alongside text prompts to capable models (OpenAI, Anthropic, Gemini, and OpenRouter models that support it). LLMPlug automatically handles fetching remote image URLs and converting them to base64 for providers that require it.
-   **JSON Mode**: Request structured JSON output from models that support it.
-   **Rich Response Metadata**: Access token usage, finish reasons, raw responses, and sometimes safety ratings.
-   **Extensive Provider Support**:
    -   **Cloud APIs:**
        -   OpenAI (GPT models)
        -   Anthropic (Claude models)
        -   Google (Gemini models)
        -   Cohere (Command models)
        -   Mistral AI (Mistral platform models)
        -   OpenRouter (Aggregator for many models, including open-source)
    -   **Local Inference (via OpenAI-Compatible APIs):**
        -   Ollama
        -   Llama.cpp (server mode)
        -   Oobabooga Text Generation WebUI (OpenAI extension)
    -   **Other:**
        -   Hugging Face (Inference API for various models - basic text features)
-   **Extensible**: Designed to easily add new providers.
-   **Error Handling**: Custom error types for better debugging.

## Installation

```bash
npm install llmplug