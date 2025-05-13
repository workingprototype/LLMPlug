import OpenAI from 'openai'; // We'll use the OpenAI SDK configured for Ollama's OpenAI-compatible endpoint
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';
import fetch from 'node-fetch'; // For Ollama-specific API calls like listing models

const OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"; // OpenAI-compatible endpoint
const OLLAMA_NATIVE_API_BASE_URL = "http://localhost:11434/api"; // For native Ollama features

export class OllamaProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "Ollama";

    // API key is not typically required for local Ollama, pass a dummy one if SDK insists
    this.apiKey = config.apiKey || 'ollama-no-key'; // Dummy key, not used by Ollama server
    
    this.baseURL = config.baseURL || OLLAMA_DEFAULT_BASE_URL;
    this.nativeBaseURL = config.nativeBaseURL || OLLAMA_NATIVE_API_BASE_URL;

    // Model is crucial for Ollama. It must be specified.
    this.defaultModel = config.defaultModel || config.model; 
    if (!this.defaultModel) {
        console.warn(`[${this.providerName}] No defaultModel specified. You'll need to provide a model for each call. Make sure the model is pulled in Ollama.`);
    }

    const openAIConfig = {
      apiKey: this.apiKey, // Will be ignored by Ollama if not configured to require one
      baseURL: this.baseURL,
      dangerouslyAllowBrowser: false,
    };

    try {
      this.client = new OpenAI(openAIConfig);
    } catch (error) {
      throw new LLMPlugConfigurationError(`Ollama (OpenAI SDK) client initialization failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * Helper to format messages for OpenAI-compatible APIs.
   * (Identical to OpenRouterProvider's _formatMessages for now)
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @returns {OpenAI.Chat.Completions.ChatCompletionMessageParam[]}
   * @protected
   */
  _formatMessages(messages) {
    return messages.map(msg => {
      let contentForAPI;
      if (Array.isArray(msg.content)) {
        contentForAPI = msg.content.map(part => {
          if (part.type === 'text') {
            return { type: 'text', text: part.text };
          } else if (part.type === 'image_url') {
            // Ollama's OpenAI-compatible endpoint might support vision models like llava
            // if the model itself supports the OpenAI vision spec.
            // We pass it through; success depends on the Ollama model.
            // For LLaVA, Ollama expects images in the 'images' array at the top level of the request,
            // not inline in content parts. This provider currently doesn't adapt to that native LLaVA/Ollama format.
            // This OpenAI-compatible path assumes the model can take image_url like GPT-4V.
            console.warn(`[${this.providerName}] Image_url content is passed through but support depends on the specific Ollama model and its OpenAI API compatibility for vision. Native Ollama LLaVA format for images is different.`);
            return { type: 'image_url', image_url: { url: part.image_url.url, detail: part.image_url.detail || 'auto' } };
          }
          return null;
        }).filter(Boolean);
        if (contentForAPI.length === 0 && msg.role !== 'assistant' && msg.role !== 'tool') {
            contentForAPI = ""; 
        } else if (contentForAPI.length === 0 && (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0)) {
            contentForAPI = null;
        }
      } else {
        contentForAPI = msg.content;
      }
      
      const apiMessage = { role: msg.role, content: contentForAPI };
      if (msg.name) apiMessage.name = msg.name;
      if (msg.tool_calls) apiMessage.tool_calls = msg.tool_calls;
      if (msg.tool_call_id) apiMessage.tool_call_id = msg.tool_call_id;
      return apiMessage;
    });
  }

  _prepareInputAsMessages(input) {
    if (typeof input === 'string') return [{ role: 'user', content: input }];
    if (Array.isArray(input)) return input;
    throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[].", this.providerName);
  }
  
  _getModel(options = {}) {
    const model = options.model || this.defaultModel;
    if (!model) {
      throw new LLMPlugConfigurationError("Model must be specified (defaultModel or per call) for Ollama. Ensure it's pulled.", this.providerName);
    }
    return model;
  }

  async generate(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    return this.chat(messages, options);
  }

  async chat(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);

    // Ollama specific options can be passed via extraParams.options
    // e.g., num_ctx, seed, stop, etc.
    // https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    // https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    const ollamaOptions = options.extraParams?.ollamaOptions || {};


    // Note on tool use with Ollama's OpenAI compatible endpoint:
    // Support depends on the model. The endpoint itself might pass tool parameters,
    // but the model needs to be fine-tuned or instructed to generate tool calls in OpenAI format.
    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens, // Ollama calls this 'num_predict' in native API, but OpenAI SDK maps max_tokens
      stop: options.stopSequences,   // Maps to 'stop' parameter in Ollama
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat, // For JSON mode, if model supports it
      top_p: options.extraParams?.topP,
      stream: false,
      options: { // Native Ollama parameters go here when using OpenAI SDK with Ollama
        temperature: options.temperature, // Redundant but often set here too
        num_predict: options.maxTokens,
        top_p: options.extraParams?.topP,
        stop: options.stopSequences,
        seed: options.extraParams?.randomSeed || options.extraParams?.seed,
        num_ctx: options.extraParams?.numCtx,
        // ... other native ollama options
        ...ollamaOptions 
      }
    };
    Object.keys(requestParams.options).forEach(key => requestParams.options[key] === undefined && delete requestParams.options[key]);
    if (Object.keys(requestParams.options).length === 0) delete requestParams.options;
    Object.keys(requestParams).forEach(key => requestParams[key] === undefined && delete requestParams[key]);

    try {
      const completion = await this.client.chat.completions.create(requestParams);
      const choice = completion.choices[0];
      if (!choice) throw new LLMPlugRequestError("Ollama API returned no choices.", this.providerName, completion);
      
      const textContent = choice.message?.content?.trim() || null;
      const toolCalls = choice.message?.tool_calls?.map(call => ({
        id: call.id || `${call.function.name}-${Date.now()}`, // Ollama might not provide ID for tool calls
        type: 'function',
        function: { name: call.function.name, arguments: call.function.arguments },
      })) || [];

      // Ollama's OpenAI compatible endpoint might not return detailed token usage.
      // Native API does for /api/generate and /api/chat
      const usage = {
        promptTokens: completion.usage?.prompt_tokens, // Often 0 or null from Ollama OpenAI endpoint
        completionTokens: completion.usage?.completion_tokens, // Often the count of generated tokens
        totalTokens: completion.usage?.total_tokens,
      };
      const finishReason = choice.finish_reason?.toLowerCase();

      return { text: textContent, toolCalls: toolCalls.length > 0 ? toolCalls : undefined, usage, finishReason, rawResponse: completion };
    } catch (error) {
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      if (error.message.includes("Connection refused")) {
        errorMessage = `Connection refused. Is Ollama server running at ${this.baseURL}? ${error.message}`;
      }
      throw new LLMPlugRequestError(`Ollama API chat request failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }

  async *generateStream(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    yield* this.chatStream(messages, options);
  }

  async *chatStream(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);
    const ollamaOptions = options.extraParams?.ollamaOptions || {};

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      // max_tokens not directly used by OpenAI SDK for stream control, but good for Ollama options
      stop: options.stopSequences,
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      top_p: options.extraParams?.topP,
      stream: true,
      options: {
        temperature: options.temperature,
        num_predict: options.maxTokens, // Max tokens for the whole generation in stream
        top_p: options.extraParams?.topP,
        stop: options.stopSequences,
        seed: options.extraParams?.randomSeed || options.extraParams?.seed,
        num_ctx: options.extraParams?.numCtx,
        ...ollamaOptions
      }
    };
    Object.keys(requestParams.options).forEach(key => requestParams.options[key] === undefined && delete requestParams.options[key]);
    if (Object.keys(requestParams.options).length === 0) delete requestParams.options;
    Object.keys(requestParams).forEach(key => requestParams[key] === undefined && delete requestParams[key]);


    try {
      const stream = await this.client.chat.completions.create(requestParams);
      let currentToolCallsState = {};

      for await (const chunk of stream) {
        const choice = chunk.choices[0];
        if (!choice) continue;

        const delta = choice.delta;
        const finishReason = choice.finish_reason?.toLowerCase();
        const chunkData = { rawChunk: chunk };

        if (delta?.content) chunkData.text = delta.content;

        if (delta?.tool_calls) {
          const processedToolCalls = [];
          for (const tcDelta of delta.tool_calls) {
            const index = tcDelta.index;
            if (tcDelta.id) {
              currentToolCallsState[index] = { id: tcDelta.id, type: 'function', function: { name: tcDelta.function?.name || '', arguments: tcDelta.function?.arguments || '' }};
            } else if (currentToolCallsState[index] && tcDelta.function) {
              if (tcDelta.function.name) currentToolCallsState[index].function.name = tcDelta.function.name;
              if (tcDelta.function.arguments) currentToolCallsState[index].function.arguments += tcDelta.function.arguments;
            }
            if(currentToolCallsState[index]) processedToolCalls.push({ ...currentToolCallsState[index] });
          }
          if (processedToolCalls.length > 0) chunkData.toolCalls = processedToolCalls;
        }
        
        if (finishReason) {
            chunkData.finishReason = finishReason;
            currentToolCallsState = {}; 
             // For Ollama, the final "usage" or metrics come from the 'done' event in native API stream
             // The OpenAI compatible stream might not provide it, or it's in the last chunk non-delta part.
             // The raw chunk for Ollama's native stream format would have `eval_count`, `eval_duration` etc. on `done:true`
             // We try to get it from `chunk.x_ollama_meta` if OpenAI SDK surfaces it or `completion.usage` if it was the last part.
            if (chunk.x_ollama_meta) { // Hypothetical field, check actual SDK output if it provides native stats
                chunkData.usage = {
                    promptTokens: chunk.x_ollama_meta.prompt_eval_count,
                    completionTokens: chunk.x_ollama_meta.eval_count,
                    // totalTokens: undefined, // Needs to be calculated or is not provided
                };
            } else if (chunk.usage) { // Standard OpenAI SDK final usage object
                 chunkData.usage = chunk.usage;
            }
        }
        yield chunkData;
      }
    } catch (error) {
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      if (error.message.includes("Connection refused")) {
        errorMessage = `Connection refused. Is Ollama server running at ${this.baseURL}? ${error.message}`;
      }
      throw new LLMPlugRequestError(`Ollama API chat stream failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }

  // --- Ollama-specific methods (optional additions) ---

  /**
   * Lists models available locally in Ollama.
   * Uses Ollama's native API.
   * @returns {Promise<string[]>} Array of model names.
   */
  async listLocalModels() {
    try {
      const response = await fetch(`${this.nativeBaseURL}/tags`);
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`Failed to list Ollama models (status ${response.status}): ${errorBody}`);
      }
      const data = await response.json();
      return data.models.map(model => model.name);
    } catch (error) {
      throw new LLMPlugRequestError(`Failed to list Ollama models: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * Pulls a model into Ollama.
   * Uses Ollama's native API.
   * @param {string} modelName - The name of the model to pull (e.g., "llama3:8b", "mistral:latest").
   * @param {boolean} [stream=false] - Whether to stream progress.
   * @returns {Promise<any | AsyncIterable<any>>} Status or stream of progress.
   */
  async pullModel(modelName, stream = false) {
    try {
      const response = await fetch(`${this.nativeBaseURL}/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName, stream: stream }),
      });
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`Failed to pull Ollama model ${modelName} (status ${response.status}): ${errorBody}`);
      }
      if (stream) {
        // Need to adapt this to an AsyncIterable<string> or similar for progress
        // For now, let's return the raw stream body if user wants to handle it.
        // Or parse line by line.
        console.warn(`[${this.providerName}] Streaming pull progress requires custom handling of the response body stream.`);
        return response.body; // User needs to handle this ReadableStream
      }
      return await response.json(); // Final status
    } catch (error) {
      throw new LLMPlugRequestError(`Failed to pull Ollama model ${modelName}: ${error.message}`, this.providerName, error);
    }
  }
}