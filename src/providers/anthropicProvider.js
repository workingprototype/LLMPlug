import Anthropic from '@anthropic-ai/sdk';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

export class AnthropicProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "Anthropic";
    try {
      this.apiKey = this._getApiKey('ANTHROPIC_API_KEY');
      this.client = new Anthropic({ apiKey: this.apiKey });
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error;
      throw new LLMPlugConfigurationError(`Anthropic client initialization failed: ${error.message}`, this.providerName, error);
    }
    this.defaultModel = config.defaultModel || 'claude-3-haiku-20240307';
  }

  /**
   * Prepares messages for the Anthropic API.
   * - Handles system prompts.
   * - Converts LLMPlug's ChatMessage format to Anthropic's format.
   * - Automatically fetches and base64 encodes image URLs.
   * - Formats tool calls and tool responses.
   * @param {import('../baseProvider.js').ChatMessage[]} messages - Array of chat messages.
   * @returns {Promise<{ systemPrompt?: string, anthropicMessages: Anthropic.Messages.MessageParam[] }>}
   * @protected
   */
  async _prepareMessages(messages) {
    let systemPrompt;
    const anthropicMessages = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        // System prompt content should be a simple string for Anthropic.
        if (typeof msg.content === 'string') {
          systemPrompt = msg.content;
        } else if (Array.isArray(msg.content) && msg.content.length > 0 && msg.content[0].type === 'text') {
          systemPrompt = msg.content.map(c => c.text || '').join('\n'); // Concatenate text parts if system prompt is array
        }
        continue; // System prompt handled, move to next message
      }

      // Process user, assistant, and tool messages
      const role = msg.role === 'tool' ? 'user' : msg.role; // Anthropic tool results are user messages
      const contentBlocks = [];

      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === 'text') {
            contentBlocks.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url') {
            // LLMPlug AUTOMATICALLY handles fetching and base64 encoding here
            const { base64Data, mimeType } = await this._fetchAndBase64Image(part.image_url.url);
            contentBlocks.push({
              type: 'image',
              source: {
                type: 'base64',
                media_type: mimeType,
                data: base64Data,
              }
            });
          } else if (part.type === 'tool_output' && msg.role === 'tool') {
            // This content part is for a 'tool' role message. It's handled by the tool_result structure below.
            // We'll add it as a specific tool_result block.
            // Anthropic expects tool_result content to be a string or JSON object.
            // For simplicity, we'll stringify if it's not already a string.
            const outputContent = typeof part.content === 'string' ? part.content : JSON.stringify(part.content);
            contentBlocks.push({
              type: 'tool_result',
              tool_use_id: msg.tool_call_id,
              content: outputContent,
              // is_error: part.is_error, // Optional: if you add is_error to ToolOutputContent
            });
          } else if (part.type === 'tool_code' ) {
             // This is usually part of an assistant message, indicating tool call arguments.
             // Anthropic handles this via the 'tool_use' block for assistant, not as separate content part usually.
             // For now, if it exists, we'll add its text.
             contentBlocks.push({ type: 'text', text: part.text });
          }
        }
      } else if (typeof msg.content === 'string') {
        contentBlocks.push({ type: 'text', text: msg.content });
      }

      // Handle assistant's decision to use tools
      if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
        msg.tool_calls.forEach(tc => {
          contentBlocks.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments), // Anthropic expects parsed JSON object for input
          });
        });
      }
      
      // Ensure that if the role is 'tool', contentBlocks should primarily contain 'tool_result'
      if (msg.role === 'tool') {
        const toolResultBlock = contentBlocks.find(cb => cb.type === 'tool_result');
        if (!toolResultBlock) {
             throw new LLMPlugRequestError(`Tool message (role 'tool') with tool_call_id '${msg.tool_call_id}' must contain a 'tool_output' content part.`, this.providerName);
        }
        // For 'tool' role, Anthropic expects the message role to be 'user' and content to be the tool_result blocks.
        anthropicMessages.push({ role: 'user', content: [toolResultBlock] });

      } else if (contentBlocks.length > 0) {
        // For user or assistant messages with content (text, image, or tool_use intent)
        anthropicMessages.push({ role: role, content: contentBlocks });
      } else if (msg.role === 'assistant' && (!msg.tool_calls || msg.tool_calls.length === 0) && msg.content === null) {
        // Assistant message with no text and no tool_calls (e.g. if only finish_reason was 'tool_calls' but calls array was empty somehow)
        // This is unusual but we can represent it as an empty content assistant message.
        anthropicMessages.push({ role: 'assistant', content: [] });
      }
    }

    if (!systemPrompt && anthropicMessages.length === 0) {
      throw new LLMPlugRequestError("Anthropic API requires at least one user message or a system prompt to proceed.", this.providerName);
    }

    return { systemPrompt, anthropicMessages };
  }

  /**
   * Converts a simple prompt string or an array of ChatMessages into a valid ChatMessage array.
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @returns {import('../baseProvider.js').ChatMessage[]}
   * @protected
   */
  _prepareInputAsMessages(input) {
    if (typeof input === 'string') {
      return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input)) {
      return input; // Assume it's already in ChatMessage[] format
    }
    throw new LLMPlugRequestError("Invalid input type. Must be a string or an array of ChatMessage objects.", this.providerName);
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
    const { systemPrompt, anthropicMessages } = await this._prepareMessages(messages);

    const requestParams = {
      model: model,
      messages: anthropicMessages,
      max_tokens: options.maxTokens || 1024, // Anthropic requires max_tokens
      temperature: options.temperature,
      stop_sequences: options.stopSequences,
      tools: options.tools?.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        input_schema: tool.function.parameters, // Anthropic uses input_schema
      })),
      ...(systemPrompt && { system: systemPrompt }), // Conditionally add system prompt
      ...options.extraParams,
    };

    try {
      const response = await this.client.messages.create(requestParams);

      const textContent = response.content.filter(block => block.type === 'text').map(block => block.text).join('').trim() || null;
      const toolCalls = response.content
        .filter(block => block.type === 'tool_use')
        .map(block => ({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input || {}), // Ensure input is stringified
          },
        }));

      const usage = {
        promptTokens: response.usage?.input_tokens,
        completionTokens: response.usage?.output_tokens,
        totalTokens: (response.usage?.input_tokens || 0) + (response.usage?.output_tokens || 0),
      };
      
      // Anthropic's stop_reason maps to finishReason
      // e.g., "end_turn", "max_tokens", "stop_sequence", "tool_use"
      const finishReason = response.stop_reason; 

      return {
        text: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: usage,
        finishReason: finishReason,
        rawResponse: response,
      };
    } catch (error) {
      const errorMessage = error.error?.message || error.message || "Unknown Anthropic API error";
      throw new LLMPlugRequestError(`Anthropic API chat request failed: ${errorMessage}`, this.providerName, error);
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
    const { systemPrompt, anthropicMessages } = await this._prepareMessages(messages);

    const requestParams = {
      model: model,
      messages: anthropicMessages,
      max_tokens: options.maxTokens || 1024,
      temperature: options.temperature,
      stop_sequences: options.stopSequences,
      tools: options.tools?.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        input_schema: tool.function.parameters,
      })),
      stream: true,
      ...(systemPrompt && { system: systemPrompt }),
      ...options.extraParams,
    };

    try {
      const stream = this.client.messages.stream(requestParams); // Use .stream() for Anthropic SDK

      // Accumulators for tool call arguments if they stream partially
      // (Anthropic usually sends tool_use input fully in content_block_start)
      const streamingToolCallArgs = {};

      for await (const event of stream) {
        const chunkData = { rawChunk: event };

        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          chunkData.text = event.delta.text;
        } else if (event.type === 'content_block_start' && event.delta?.type === 'input_json_delta') {
            // This event type is for Claude 3.5 Sonnet streaming tool inputs
            if (event.content_block.type === 'tool_use') {
                const toolUseId = event.content_block.id;
                if (!streamingToolCallArgs[toolUseId]) {
                    streamingToolCallArgs[toolUseId] = {
                        id: toolUseId,
                        name: event.content_block.name,
                        arguments: ''
                    };
                }
                streamingToolCallArgs[toolUseId].arguments += event.delta.partial_json;
            }
        } else if (event.type === 'content_block_start' && event.content_block.type === 'tool_use') {
          // For models older than Claude 3.5 Sonnet, input comes fully here
          chunkData.toolCalls = [{
            id: event.content_block.id,
            type: 'function',
            function: {
              name: event.content_block.name,
              arguments: JSON.stringify(event.content_block.input || {}),
            }
          }];
        } else if (event.type === 'message_delta' && event.delta.stop_reason) {
            // Claude 3.5 Sonnet sends stop_reason and usage in message_delta
            chunkData.finishReason = event.delta.stop_reason;
            if (event.usage) { // Check if usage is present on this specific event
                chunkData.usage = {
                    output_tokens: event.usage.output_tokens,
                    // input_tokens usually comes in message_start
                };
            }
        } else if (event.type === 'message_start') {
            // Contains input_tokens
            if (event.message.usage) {
                 chunkData.usage = { promptTokens: event.message.usage.input_tokens };
            }
        } else if (event.type === 'message_stop') {
          // This event signals the end of the stream.
          // For Claude 3.5 Sonnet, tool_calls derived from input_json_delta should be finalized here
          const finalizedToolCalls = Object.values(streamingToolCallArgs).map(tc => ({
                id: tc.id,
                type: 'function',
                function: { name: tc.name, arguments: tc.arguments }
          }));
          if (finalizedToolCalls.length > 0) {
              chunkData.toolCalls = finalizedToolCalls;
          }
          
          // Older models might send final usage/stop_reason here.
          // For Claude 3.5 Sonnet, these are in message_delta.
          // We need to check the Anthropic API documentation for which models use which event types for final info.
          // Let's assume `message_delta` is primary for new models.
        }
        yield chunkData;
      }
    } catch (error) {
      const errorMessage = error.error?.message || error.message || "Unknown Anthropic API stream error";
      throw new LLMPlugRequestError(`Anthropic API chat stream failed: ${errorMessage}`, this.providerName, error);
    }
  }
}