// This acts as a base for local servers that expose an OpenAI-compatible API.
// It's very similar to OpenRouterProvider and OllamaProvider in its use of the OpenAI SDK.

import OpenAI from 'openai';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

export class GenericOpenAICompatibleProvider extends BaseProvider {
  constructor(config = {}, providerName = "GenericOpenAICompatible", defaultBaseURL = "http://localhost:8000/v1") {
    super(config);
    this.providerName = providerName;

    // API key is often not required or is a fixed string for local servers
    this.apiKey = config.apiKey || 'local-no-key'; 
    
    this.baseURL = config.baseURL || defaultBaseURL;

    this.defaultModel = config.defaultModel || config.model; 
    if (!this.defaultModel) {
        console.warn(`[${this.providerName}] No defaultModel specified. You'll need to provide a model for each call. This model name must match what the local server expects.`);
    }

    const openAIConfig = {
      apiKey: this.apiKey,
      baseURL: this.baseURL,
      dangerouslyAllowBrowser: false, // Server-side
      ...(config.sdkConfig || {}) // Allow passing further OpenAI SDK config options
    };

    try {
      this.client = new OpenAI(openAIConfig);
    } catch (error) {
      throw new LLMPlugConfigurationError(`[${this.providerName}] OpenAI SDK client initialization failed: ${error.message}`, this.providerName, error);
    }
  }

  // Re-use message formatting, input prep, and model getter logic
  // (These can be identical to OpenRouterProvider or OllamaProvider's versions)
  _formatMessages(messages) {
    return messages.map(msg => {
      let contentForAPI;
      if (Array.isArray(msg.content)) {
        contentForAPI = msg.content.map(part => {
          if (part.type === 'text') return { type: 'text', text: part.text };
          if (part.type === 'image_url') {
            console.warn(`[${this.providerName}] Image_url content passed through. Support depends on the local model/server's OpenAI API compatibility for vision.`);
            return { type: 'image_url', image_url: { url: part.image_url.url, detail: part.image_url.detail || 'auto' } };
          }
          return null;
        }).filter(Boolean);
        if (contentForAPI.length === 0 && msg.role !== 'assistant' && msg.role !== 'tool') contentForAPI = ""; 
        else if (contentForAPI.length === 0 && (msg.role === 'assistant' && msg.tool_calls?.length > 0)) contentForAPI = null;
      } else contentForAPI = msg.content;
      
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
    throw new LLMPlugRequestError("Invalid input type. Must be string or ChatMessage[].", this.providerName);
  }
  
  _getModel(options = {}) {
    const model = options.model || this.defaultModel;
    if (!model) {
      throw new LLMPlugConfigurationError(`Model must be specified for ${this.providerName}. This usually corresponds to the model file loaded by the local server.`, this.providerName);
    }
    return model; // For many local servers, this might be a filename or alias configured in the server
  }

  async generate(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    return this.chat(messages, options);
  }

  async chat(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);

    // Parameters for local servers can vary widely in what they respect.
    // Some might only respect temp/max_tokens.
    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stop: options.stopSequences,
      tools: options.tools, // Support depends heavily on the local server & model
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      top_p: options.extraParams?.topP,
      // Some local servers might accept other OpenAI params or their own custom ones via extraParams
      ...(options.extraParams?.serverSpecificParams || {}), // For truly server-specific params
      ...options.extraParams,
    };
    delete requestParams.serverSpecificParams; // clean up
    Object.keys(requestParams).forEach(key => requestParams[key] === undefined && delete requestParams[key]);

    try {
      const completion = await this.client.chat.completions.create(requestParams);
      const choice = completion.choices[0];
      if (!choice) throw new LLMPlugRequestError(`[${this.providerName}] API returned no choices for model ${model}.`, this.providerName, completion);
      
      const textContent = choice.message?.content?.trim() || null;
      const toolCalls = choice.message?.tool_calls?.map(call => ({
        id: call.id || `${call.function.name}-${Date.now()}`,
        type: 'function',
        function: { name: call.function.name, arguments: call.function.arguments },
      })) || [];

      const usage = { // Often not accurately reported by local servers
        promptTokens: completion.usage?.prompt_tokens,
        completionTokens: completion.usage?.completion_tokens,
        totalTokens: completion.usage?.total_tokens,
      };
      const finishReason = choice.finish_reason?.toLowerCase();

      return { text: textContent, toolCalls: toolCalls.length > 0 ? toolCalls : undefined, usage, finishReason, rawResponse: completion };
    } catch (error) {
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      if (error.message.includes("Connection refused")) {
        errorMessage = `Connection refused. Is ${this.providerName} server running at ${this.baseURL}? ${error.message}`;
      }
      throw new LLMPlugRequestError(`[${this.providerName}] API chat request failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }

  async *generateStream(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    yield* this.chatStream(messages, options);
  }

  async *chatStream(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens, // Max tokens might be advisory for stream control on some local servers
      stop: options.stopSequences,
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      top_p: options.extraParams?.topP,
      stream: true,
      ...(options.extraParams?.serverSpecificParams || {}),
      ...options.extraParams,
    };
    delete requestParams.serverSpecificParams;
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
        }
        if (chunk.usage) chunkData.usage = chunk.usage; // Usage in last stream chunk if available

        yield chunkData;
      }
    } catch (error) {
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      if (error.message.includes("Connection refused")) {
        errorMessage = `Connection refused. Is ${this.providerName} server running at ${this.baseURL}? ${error.message}`;
      }
      throw new LLMPlugRequestError(`[${this.providerName}] API chat stream failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }
}