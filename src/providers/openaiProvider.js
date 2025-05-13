import OpenAI from 'openai';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

export class OpenAIProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "OpenAI";
    try {
      this.apiKey = this._getApiKey('OPENAI_API_KEY');
      this.client = new OpenAI({ apiKey: this.apiKey });
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error;
      throw new LLMPlugConfigurationError(`OpenAI client initialization failed: ${error.message}`, this.providerName, error);
    }
    this.defaultModel = config.defaultModel || 'gpt-3.5-turbo';
  }

  /**
   * Helper to format messages for OpenAI.
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @returns {OpenAI.Chat.Completions.ChatCompletionMessageParam[]}
   */
  _formatMessagesForOpenAI(messages) {
    return messages.map(msg => {
      const formattedContent = this._normalizeContent(msg.content);
      const openAIMessage = {
        role: msg.role,
        content: formattedContent
      };
      if (msg.name) openAIMessage.name = msg.name;
      if (msg.tool_calls) openAIMessage.tool_calls = msg.tool_calls;
      if (msg.tool_call_id) openAIMessage.tool_call_id = msg.tool_call_id;

      return openAIMessage;
    });
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
      // Ensure proper formatting for multi-modal content if present.
      // The _formatMessagesForOpenAI will handle the normalization.
      return input;
    }
    throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[]", this.providerName);
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async generate(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    return this.chat(messages, options);
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async chat(messages, options = {}) {
    const model = options.model || this.defaultModel;
    const formattedMessages = this._formatMessagesForOpenAI(messages);

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stop: options.stopSequences,
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      ...options.extraParams,
    };

    try {
      const completion = await this.client.chat.completions.create(requestParams);

      const textContent = completion.choices[0]?.message?.content?.trim() || '';
      const toolCalls = completion.choices[0]?.message?.tool_calls?.map(call => ({
        id: call.id,
        type: call.type,
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
      const finishReason = completion.choices[0]?.finish_reason;

      return {
        text: textContent,
        toolCalls: toolCalls,
        usage: usage,
        finishReason: finishReason,
        rawResponse: completion,
      };
    } catch (error) {
      throw new LLMPlugRequestError(`OpenAI API chat request failed: ${error.message}`, this.providerName, error);
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
    const model = options.model || this.defaultModel;
    const formattedMessages = this._formatMessagesForOpenAI(messages);

    const requestParams = {
      model: model,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stop: options.stopSequences,
      tools: options.tools,
      tool_choice: options.toolChoice,
      response_format: options.responseFormat,
      stream: true, // Crucial for streaming
      ...options.extraParams,
    };

    try {
      const stream = await this.client.chat.completions.create(requestParams);

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
        const finishReason = chunk.choices[0]?.finish_reason;
        const usage = chunk.usage; // Usage typically comes at the end of the stream

        const chunkData = { rawChunk: chunk };

        if (delta?.content) {
          chunkData.text = delta.content;
        }

        // Handle tool calls streaming
        if (delta?.tool_calls) {
          chunkData.toolCalls = delta.tool_calls.map(call => ({
            id: call.id,
            type: call.type,
            function: {
              name: call.function.name,
              arguments: call.function.arguments || '', // arguments come as chunks, so accumulate later
            },
          }));
        }
        
        if (finishReason) {
          chunkData.finishReason = finishReason;
        }
        if (usage) { // Full usage object usually comes on the final chunk
            chunkData.usage = {
                promptTokens: usage.prompt_tokens,
                completionTokens: usage.completion_tokens,
                totalTokens: usage.total_tokens,
            };
        }

        yield chunkData;
      }
    } catch (error) {
      throw new LLMPlugRequestError(`OpenAI API chat stream failed: ${error.message}`, this.providerName, error);
    }
  }
}