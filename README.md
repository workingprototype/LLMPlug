# LLMPlug

LLMPlug is a Node.js library designed to simplify the integration of various Large Language Models (LLMs) from different vendors into your applications. It provides a unified interface for common LLM tasks like text generation and chat completions, now with advanced features.

## Features

-   **Unified API**: Use the same methods (`generate`, `chat`, `generateStream`, `chatStream`) for different LLM providers.
-   **Easy Configuration**: Simple setup for API keys and model preferences.
-   **Streaming Responses**: Get partial responses as they are generated, improving user experience.
-   **Function Calling (Tool Use)**: Enable LLMs to interact with external tools and APIs (e.g., retrieve real-time data, execute code).
-   **Multimodal Input**: Send images alongside text prompts to capable models (e.g., GPT-4o, Gemini Vision, Claude 3 Vision). **LLMPlug automatically handles fetching remote image URLs and converting them to base64 for providers that require it.**
-   **JSON Mode**: Request structured JSON output from models that support it.
-   **Rich Response Metadata**: Access token usage, finish reasons, and raw responses.
-   **Provider Support**:
    -   OpenAI (GPT models) - Full support for all features.
    -   Anthropic (Claude models) - Streaming, Multimodal (Base64 only for now), Tool Use.
    -   Google (Gemini models) - Streaming, Multimodal (Base64 only for now), Tool Use.
    -   Hugging Face (Inference API for various models) - Basic text generation/chat; streaming, tool use, and multimodal input are **not** generically supported across all models via the Inference API.
-   **Extensible**: Designed to easily add new providers.
-   **Error Handling**: Custom error types for better debugging.

## Installation

```bash
npm install llmplug