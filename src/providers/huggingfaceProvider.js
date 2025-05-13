import fetch from 'node-fetch';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

const HUGGINGFACE_API_BASE_URL = "https://api-inference.huggingface.co/models/";

export class HuggingFaceProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "HuggingFace";
    try {
      this.apiToken = this._getApiKey('HUGGINGFACE_API_TOKEN', 'apiToken');
    } catch (error) {
      console.warn("HuggingFace API token not found. For private models or higher rate limits, please provide one.");
      this.apiToken = null;
    }
    if (!config.modelId) {
      throw new LLMPlugConfigurationError("`modelId` is required in config for HuggingFaceProvider (e.g., 'gpt2', 'mistralai/Mistral-7B-Instruct-v0.1').", this.providerName);
    }
    this.modelId = config.modelId;
    this.task = config.task || 'text-generation'; // or 'conversational'
  }

  async _makeApiCall(payload, modelIdOverride = null, taskOverride = null) {
    const effectiveModelId = modelIdOverride || this.modelId;
    const effectiveTask = taskOverride || this.task;
    const apiUrl = `${HUGGINGFACE_API_BASE_URL}${effectiveModelId}`;

    const headers = {
      'Content-Type': 'application/json',
    };
    if (this.apiToken) {
      headers['Authorization'] = `Bearer ${this.apiToken}`;
    }

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new LLMPlugRequestError(
          `Hugging Face API request failed for model ${effectiveModelId} with status ${response.status}: ${errorBody}`,
          this.providerName
        );
      }
      return await response.json();
    } catch (error) {
      if (error instanceof LLMPlugRequestError) throw error;
      throw new LLMPlugRequestError(`Hugging Face API request error: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * Converts a prompt string into the chat message array expected by chat API.
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @returns {import('../baseProvider.js').ChatMessage[]}
   */
  _prepareInputAsMessages(input) {
    if (typeof input === 'string') {
      return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input)) {
      // HuggingFace Inference API is generally text-only unless a specific model is used.
      // For simplicity, we'll only extract text content.
      return input.map(msg => ({
        role: msg.role,
        content: Array.isArray(msg.content) ? msg.content.map(part => part.type === 'text' ? part.text : '').join('') : msg.content,
      }));
    }
    throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[]", this.providerName);
  }


  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async generate(input, options = {}) {
    // For Hugging Face, `generate` will typically map to the `text-generation` task.
    // If input is messages, we'll convert it to a single prompt string.
    let prompt;
    if (Array.isArray(input)) {
        // Simplistic concatenation for chat messages into a single prompt string
        prompt = input.map(msg => `${msg.role}: ${typeof msg.content === 'string' ? msg.content : msg.content.map(p => p.text).join('\n')}`).join('\n') + '\nassistant:';
    } else {
        prompt = input;
    }

    const modelId = options.model || this.modelId;
    const payload = {
      inputs: prompt,
      parameters: {
        max_new_tokens: options.maxTokens,
        temperature: options.temperature,
        return_full_text: false,
        stop_sequences: options.stopSequences,
        ...(options.extraParams || {}),
      },
      options: {
        wait_for_model: true,
        use_cache: options.extraParams?.use_cache !== undefined ? options.extraParams.use_cache : true,
      }
    };

    try {
      const apiResponse = await this._makeApiCall(payload, modelId, 'text-generation');
      let textContent = '';
      if (Array.isArray(apiResponse) && apiResponse.length > 0 && apiResponse[0].generated_text) {
        textContent = apiResponse[0].generated_text.trim();
      } else if (apiResponse && apiResponse.generated_text) {
        textContent = apiResponse.generated_text.trim();
      } else {
        console.warn("HuggingFace generate: Unexpected response format", apiResponse);
      }

      // Hugging Face Inference API typically does not provide token usage directly for all models.
      // Finish reason is also not standardized.
      return {
        text: textContent,
        usage: null, // Not available in standard Inference API response
        finishReason: null, // Not available in standard Inference API response
        rawResponse: apiResponse,
      };
    } catch (error) {
      throw new LLMPlugRequestError(`Hugging Face API generate request failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async chat(messages, options = {}) {
    const modelId = options.model || this.modelId;
    const pastUserInputs = [];
    const generatedResponses = [];
    let currentQuery = ""; // This will be the last user message
    let systemPrompt = "";

    messages.forEach(msg => {
      const contentText = typeof msg.content === 'string' ? msg.content : msg.content.map(p => p.type === 'text' ? p.text : '').join('');

      if (msg.role === 'system') {
        systemPrompt += (systemPrompt ? "\n" : "") + contentText;
      } else if (msg.role === 'user') {
        // If the last message was user, merge it (simplistic, ideally new turn)
        if (currentQuery) {
          pastUserInputs.push(currentQuery);
          generatedResponses.push(""); // No assistant response yet for this past user input
        }
        currentQuery = contentText;
      } else if (msg.role === 'assistant') {
        // This implies a response to `currentQuery`. If currentQuery is empty, it's history.
        if (currentQuery) { // This assistant message is a response to `currentQuery`
            pastUserInputs.push(currentQuery);
            generatedResponses.push(contentText);
            currentQuery = ""; // Reset for next user message
        } else { // This is part of historical conversation without a preceding user message in this chunk.
            // If history already has an assistant, this implies malformed history.
            // For simplicity, we'll just push it.
            generatedResponses.push(contentText);
            if (pastUserInputs.length < generatedResponses.length) {
                pastUserInputs.push(""); // Add a placeholder user input if missing
            }
        }
      }
    });

    // Prepend system prompt to the final user query if any
    if (systemPrompt) {
        currentQuery = systemPrompt + "\n" + currentQuery;
    }

    if (!currentQuery) {
        // If no new user query, maybe the intent was to just pass history.
        // But for conversational API, a new query is usually expected.
        throw new LLMPlugRequestError("HuggingFace chat: No current user message provided to generate a response for.", this.providerName);
    }


    const payload = {
      inputs: {
        text: currentQuery, // The current user input
        past_user_inputs: pastUserInputs,
        generated_responses: generatedResponses,
      },
      parameters: {
        max_new_tokens: options.maxTokens,
        temperature: options.temperature,
        stop_sequences: options.stopSequences,
        ...(options.extraParams || {}),
      },
      options: {
        wait_for_model: true,
        use_cache: options.extraParams?.use_cache !== undefined ? options.extraParams.use_cache : true,
      }
    };

    try {
      const apiResponse = await this._makeApiCall(payload, modelId, 'conversational');

      let textContent = '';
      if (apiResponse && apiResponse.generated_text) {
        textContent = apiResponse.generated_text.trim();
      } else if (apiResponse && apiResponse.conversation && apiResponse.conversation.generated_responses) {
        const newResponses = apiResponse.conversation.generated_responses;
        if (newResponses.length > generatedResponses.length) {
          textContent = newResponses[newResponses.length - 1].trim();
        }
      } else {
        console.warn("HuggingFace chat: Unexpected response format", apiResponse);
      }

      return {
        text: textContent,
        usage: null,
        finishReason: null,
        rawResponse: apiResponse,
      };
    } catch (error) {
      throw new LLMPlugRequestError(`Hugging Face API chat request failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *generateStream(input, options = {}) {
    throw new LLMPlugError(`'generateStream' method not implemented for HuggingFaceProvider. Streaming is highly model-dependent for Inference API.`, this.providerName);
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *chatStream(messages, options = {}) {
    throw new LLMPlugError(`'chatStream' method not implemented for HuggingFaceProvider. Streaming is highly model-dependent for Inference API.`, this.providerName);
  }
}