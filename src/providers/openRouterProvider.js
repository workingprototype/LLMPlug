import OpenAI from 'openai'; // We'll use the OpenAI SDK configured for OpenRouter
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

const OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1";

export class OpenRouterProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "OpenRouter";

    try {
      this.apiKey = this._getApiKey('OPENROUTER_API_KEY');
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error; // Re-throw config errors
      // If API key is not found, it's a fatal error for this provider.
      throw new LLMPlugConfigurationError(`OpenRouter API key (OPENROUTER_API_KEY) is required.`, this.providerName);
    }

    // Model is crucial for OpenRouter. It can be a default or passed per call.
    // Format: "vendor/model" e.g., "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"
    this.defaultModel = config.defaultModel; // User can set a default model in config
    if (!this.defaultModel && !config.model) { // model can be an alias for defaultModel
        console.warn(`[${this.providerName}] No defaultModel specified. You'll need to provide a model for each call.`);
    }
    this.defaultModel = this.defaultModel || config.model;


    const openAIConfig = {
      apiKey: this.apiKey,
      baseURL: config.baseURL || OPENROUTER_API_BASE_URL,
      defaultHeaders: {
        // OpenRouter recommends these headers
        'HTTP-Referer': config.httpReferer || 'https://llmplug.dev', // Replace with your actual site/app URL
        'X-Title': config.xTitle || 'LLMPlug Application',        // Replace with your app name
        ...(config.extraHeaders || {}),
      },
      dangerouslyAllowBrowser: false, // Ensure this is false for server-side usage
    };

    try {
      this.client = new OpenAI(openAIConfig);
    } catch (error) {
      throw new LLMPlugConfigurationError(`OpenRouter (OpenAI SDK) client initialization failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * Helper to format messages for OpenAI-compatible APIs.
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
            // OpenRouter passes image_url through to underlying OpenAI-compatible vision models
            return { type: 'image_url', image_url: { url: part.image_url.url, detail: part.image_url.detail || 'auto' } };
          }
          // Other content types like tool_code/tool_output are not directly part of message content array for OpenAI spec
          console.warn(`[${this.providerName}] Unsupported content part type '${part.type}' in message content array.`);
          return null;
        }).filter(Boolean);
        // If all parts were filtered (e.g. only unsupported types), use an empty string or handle error
        if (contentForAPI.length === 0 && msg.role !== 'assistant' && msg.role !== 'tool') {
            contentForAPI = ""; // Avoid sending empty array if it's not an assistant/tool message
        } else if (contentForAPI.length === 0 && (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0)) {
            contentForAPI = null; // OpenAI spec allows null content for assistant with tool_calls
        }

      } else {
        contentForAPI = msg.content; // Assumed string
      }
      
      const apiMessage = {
        role: msg.role,
        content: contentForAPI,
      };

      if (msg.name) apiMessage.name = msg.name;
      if (msg.tool_calls) apiMessage.tool_calls = msg.tool_calls; // Already in OpenAI format
      if (msg.tool_call_id) apiMessage.tool_call_id = msg.tool_call_id;

      return apiMessage;
    });
  }

  /**
   * Converts a prompt string or ChatMessage array into a ChatMessage array.
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @returns {import('../baseProvider.js').ChatMessage[]}
   * @protected
   */
  _prepareInputAsMessages(input) {
    if (typeof input === 'string') {
      return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input)) {
      return input;
    }
    throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[].", this.providerName);
  }
  
  _getModel(options = {}) {
    const model = options.model || this.defaultModel;
    if (!model) {
      throw new LLMPlugConfigurationError("Model must be specified either in provider config (defaultModel) or per call for OpenRouter.", this.providerName);
    }
    return model;
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async generate(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    // `generate` is a convenience; map to `chat` as OpenRouter is OpenAI-compatible chat endpoint
    return this.chat(messages, options);
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async chat(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stop: options.stopSequences,
      tools: options.tools, // Assumes OpenAI tool format
      tool_choice: options.toolChoice, // Assumes OpenAI tool_choice format
      response_format: options.responseFormat, // Assumes OpenAI response_format (for JSON mode)
      top_p: options.extraParams?.topP, // Common OpenAI param
      // n: options.extraParams?.n, // Number of completions (not typical for LLMPlug's single string result)
      // presence_penalty: options.extraParams?.presencePenalty,
      // frequency_penalty: options.extraParams?.frequencyPenalty,
      // logit_bias: options.extraParams?.logitBias,
      // user: options.extraParams?.user, // User ID for tracking
      ...options.extraParams, // Allow passthrough of other OpenAI compatible params
    };

    // Remove undefined params to keep payload clean
    Object.keys(requestParams).forEach(key => requestParams[key] === undefined && delete requestParams[key]);


    try {
      const completion = await this.client.chat.completions.create(requestParams);

      const choice = completion.choices[0];
      if (!choice) {
        throw new LLMPlugRequestError("OpenRouter API returned no choices.", this.providerName, completion);
      }
      
      const textContent = choice.message?.content?.trim() || null;
      const toolCalls = choice.message?.tool_calls?.map(call => ({
        id: call.id,
        type: call.type, // Should be 'function'
        function: {
          name: call.function.name,
          arguments: call.function.arguments,
        },
      })) || [];

      const usage = {
        promptTokens: completion.usage?.prompt_tokens,
        completionTokens: completion.usage?.completion_tokens,
        totalTokens: completion.usage?.total_tokens,
      };
      const finishReason = choice.finish_reason?.toLowerCase();

      return {
        text: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: usage,
        finishReason: finishReason,
        rawResponse: completion,
      };
    } catch (error) {
      // OpenAI SDK errors are usually well-structured
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      if (error.response && error.response.data && error.response.data.error && error.response.data.error.message) {
        errorMessage += ` - API Error: ${error.response.data.error.message}`;
      } else if (error.error && error.error.message) { // Sometimes error is nested differently
        errorMessage += ` - API Error: ${error.error.message}`;
      }
      throw new LLMPlugRequestError(`OpenRouter API (via OpenAI SDK) chat request failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *generateStream(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    yield* this.chatStream(messages, options);
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *chatStream(messages, options = {}) {
    const model = this._getModel(options);
    const formattedMessages = this._formatMessages(messages);

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stop: options.stopSequences,
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      top_p: options.extraParams?.topP,
      stream: true, // Crucial for streaming
      ...options.extraParams,
    };
    Object.keys(requestParams).forEach(key => requestParams[key] === undefined && delete requestParams[key]);


    try {
      const stream = await this.client.chat.completions.create(requestParams);
      let currentToolCallsState = {}; // To accumulate arguments for each tool call by index

      for await (const chunk of stream) {
        const choice = chunk.choices[0];
        if (!choice) continue; // Skip empty chunks if any

        const delta = choice.delta;
        const finishReason = choice.finish_reason?.toLowerCase();
        const chunkData = { rawChunk: chunk };

        if (delta?.content) {
          chunkData.text = delta.content;
        }

        if (delta?.tool_calls) {
          const processedToolCalls = [];
          for (const tcDelta of delta.tool_calls) {
            const index = tcDelta.index; // OpenAI SDK provides index for tool call deltas
            if (tcDelta.id) { // Start of a new tool call
              currentToolCallsState[index] = {
                id: tcDelta.id,
                type: 'function', // Assuming 'function'
                function: { name: tcDelta.function?.name || '', arguments: tcDelta.function?.arguments || '' }
              };
            } else if (currentToolCallsState[index] && tcDelta.function) { // Continuation of an existing tool call
              if (tcDelta.function.name) { // Should ideally not change after first announcement
                currentToolCallsState[index].function.name = tcDelta.function.name;
              }
              if (tcDelta.function.arguments) {
                currentToolCallsState[index].function.arguments += tcDelta.function.arguments;
              }
            }
            // Add the current state of this tool call to the chunk if it's been initialized
            if(currentToolCallsState[index]) {
                 processedToolCalls.push({ ...currentToolCallsState[index] });
            }
          }
          if (processedToolCalls.length > 0) {
            chunkData.toolCalls = processedToolCalls;
          }
        }
        
        if (finishReason) {
          chunkData.finishReason = finishReason;
          currentToolCallsState = {}; // Reset for next potential message in a multi-response stream (rare)
        }
        
        // OpenAI stream (and thus OpenRouter) usually includes usage in the *last* chunk if at all for stream.
        // Some models/endpoints might provide it.
        if (chunk.usage) { 
            chunkData.usage = {
                promptTokens: chunk.usage.prompt_tokens,
                completionTokens: chunk.usage.completion_tokens,
                totalTokens: chunk.usage.total_tokens,
            };
        }

        yield chunkData;
      }
    } catch (error) {
      let errorMessage = error.message;
      if (error.status) errorMessage = `(Status ${error.status}) ${error.message}`;
      throw new LLMPlugRequestError(`OpenRouter API (via OpenAI SDK) chat stream failed for model ${model}: ${errorMessage}`, this.providerName, error);
    }
  }
}