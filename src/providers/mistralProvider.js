import MistralClient from '@mistralai/mistralai';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError, LLMPlugToolError } from '../utils/errors.js';

export class MistralProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "MistralAI";
    try {
      this.apiKey = this._getApiKey('MISTRAL_API_KEY');
      this.client = new MistralClient(this.apiKey);
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error;
      throw new LLMPlugConfigurationError(`Mistral AI client initialization failed: ${error.message}`, this.providerName, error);
    }
    // Common models: mistral-tiny, mistral-small, mistral-medium, mistral-large-latest
    // New models: open-mistral-7b, open-mixtral-8x7b, mistral-embed
    // For chat, 'mistral-large-latest' or 'open-mixtral-8x7b' are good choices for capability.
    // 'mistral-small-latest' is a good balance.
    this.defaultModel = config.defaultModel || 'mistral-small-latest'; 
  }

  /**
   * Prepares messages for the Mistral AI API.
   * - Mistral expects an array of { role: "system" | "user" | "assistant" | "tool", content: string, tool_calls?: [], tool_call_id?: string }
   * - Multimodal content (images) is not directly supported by the main chat completion API.
   * @param {import('../baseProvider.js').ChatMessage[]} messages - Array of chat messages.
   * @returns {Promise<MistralClient.ChatMessage[]>} - Messages formatted for Mistral client.
   * @protected
   */
  async _prepareMessages(messages) {
    const mistralMessages = [];
    for (const msg of messages) {
      let contentText = "";

      if (typeof msg.content === 'string') {
        contentText = msg.content;
      } else if (Array.isArray(msg.content)) {
        // Concatenate text parts. Ignore non-text parts like images for Mistral chat.
        contentText = msg.content
          .filter(part => part.type === 'text')
          .map(part => part.text)
          .join('\n');
        
        if (msg.content.some(part => part.type === 'image_url')) {
          console.warn(`[${this.providerName}] Image content parts are ignored as the chat API is primarily text-based.`);
        }
        // If tool_output is part of content array for a 'tool' role message
        if (msg.role === 'tool' && msg.content.some(p => p.type === 'tool_output')) {
             const toolOutputPart = msg.content.find(p => p.type === 'tool_output');
             // Mistral expects content of tool message to be the stringified JSON output
             if (toolOutputPart) {
                contentText = typeof toolOutputPart.content === 'string' ? toolOutputPart.content : JSON.stringify(toolOutputPart.content);
             }
        }
      }

      const mistralMessage = {
        role: msg.role, // 'system', 'user', 'assistant', 'tool'
        content: contentText,
      };

      // Add tool_calls for assistant messages
      if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
        mistralMessage.tool_calls = msg.tool_calls.map(tc => ({
          id: tc.id, // Mistral uses 'id'
          type: tc.type, // 'function'
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments, // JSON string
          }
        }));
      }

      // Add tool_call_id for tool messages
      if (msg.role === 'tool' && msg.tool_call_id) {
        mistralMessage.tool_call_id = msg.tool_call_id;
        // The 'name' property for tool role is not standard in Mistral's ChatMessage,
        // but the content itself should be the output.
        // If msg.name was intended for the function name, it's implicitly handled by tool_call_id linkage.
      }
      
      // Mistral API might be sensitive to empty content for certain roles,
      // especially if it's not an assistant message with tool_calls.
      if (mistralMessage.content === "" && !(msg.role === 'assistant' && mistralMessage.tool_calls?.length > 0)) {
        // If content is empty and it's not an assistant message making tool calls,
        // it might be problematic. For now, we pass it as is.
        // Mistral API docs suggest content can be null for assistant messages with tool_calls.
        if (msg.role === 'assistant' && mistralMessage.tool_calls?.length > 0) {
            mistralMessage.content = null; // Explicitly set to null as per some interpretations of Mistral docs
        } else if (mistralMessage.content === "" && (msg.role === 'user' || msg.role === 'system')) {
            // console.warn(`[${this.providerName}] Sending message with empty content for role '${msg.role}'. This might be ignored or cause an error.`);
            // Mistral seems to handle empty user/system messages, but good to be aware.
        }
      }


      mistralMessages.push(mistralMessage);
    }
    return mistralMessages;
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async generate(input, options = {}) {
    // Mistral's primary interface is chat. We'll adapt `generate` to use the chat endpoint.
    let messagesForChat;
    if (typeof input === 'string') {
      messagesForChat = [{ role: 'user', content: input }];
    } else if (Array.isArray(input)) {
      messagesForChat = input;
    } else {
      throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[].", this.providerName);
    }
    // `generate` on Mistral doesn't usually involve tools or complex chat history structure,
    // so we pass minimal options from `GenerationOptions` relevant to a single completion.
    return this.chat(messagesForChat, {
        model: options.model, // Allow overriding model
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        extraParams: {
            topP: options.extraParams?.topP, // Mistral uses topP
            randomSeed: options.extraParams?.randomSeed,
            // safePrompt: options.extraParams?.safePrompt // For moderation
        }
    });
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async chat(messages, options = {}) {
    const model = options.model || this.defaultModel;
    const mistralMessages = await this._prepareMessages(messages);

    const mistralTools = options.tools?.map(tool => ({
      type: 'function',
      function: {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters, // Mistral expects JSON schema directly
      }
    }));

    let toolChoiceOption = options.toolChoice;
    if (typeof options.toolChoice === 'object' && options.toolChoice.type === 'function') {
        toolChoiceOption = `TOOL_ID:${options.toolChoice.function.name}`; // This syntax is a guess, check Mistral docs
                                                                       // Or more likely, 'function' and function name directly
        // Mistral's 'tool_choice' can be 'auto', 'any', or 'none'.
        // Forcing a specific function might be 'any' with only one tool, or might require specific formatting.
        // The SDK documentation should clarify this. For now, we'll map simply.
        // The official client library seems to take 'auto', 'any', 'none'.
        // To force a specific tool, you usually provide ONLY that tool in the `tools` array and set `tool_choice: 'any'`.
        // Let's simplify and assume 'auto', 'any', 'none' are the primary direct values for tool_choice.
        // If user passes object, it's more complex. For now, we only pass string values.
        if (typeof options.toolChoice !== 'string') {
            console.warn(`[${this.providerName}] Complex tool_choice object not directly supported. Using 'auto'. Provide 'auto', 'any', or 'none'.`);
            toolChoiceOption = 'auto';
        }
    }


    try {
      const response = await this.client.chat({
        model: model,
        messages: mistralMessages,
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        topP: options.extraParams?.topP,
        randomSeed: options.extraParams?.randomSeed,
        // safePrompt: options.extraParams?.safePrompt, // Moderation
        tools: mistralTools,
        toolChoice: typeof toolChoiceOption === 'string' ? toolChoiceOption : 'auto', // 'auto', 'any', 'none'
        responseFormat: options.responseFormat?.type === 'json_object' ? { type: 'json_object' } : undefined, // For JSON mode
      });

      const choice = response.choices[0];
      if (!choice) {
        throw new LLMPlugRequestError("Mistral API returned no choices.", this.providerName, response);
      }

      const textContent = choice.message.content?.trim() || null;
      const toolCalls = choice.message.tool_calls?.map(tc => ({
        id: tc.id,
        type: 'function', // Assuming 'function' as Mistral uses this type
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments, // Already a JSON string
        },
      })) || [];

      const usage = {
        promptTokens: response.usage?.prompt_tokens,
        completionTokens: response.usage?.completion_tokens,
        totalTokens: response.usage?.total_tokens,
      };
      
      // Mistral finish reasons: "stop", "length", "tool_calls", "error", "other"
      const finishReason = choice.finish_reason?.toLowerCase();

      return {
        text: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: usage,
        finishReason: finishReason,
        rawResponse: response,
      };
    } catch (error) {
      // The Mistral client often throws errors with useful `status` and `message`
      let errorMessage = error.message;
      if (error.status) {
          errorMessage = `(Status ${error.status}) ${error.message}`;
      }
      if (error.response && error.response.data && error.response.data.message) {
        errorMessage += ` - API Message: ${error.response.data.message}`;
      }
      throw new LLMPlugRequestError(`Mistral AI API chat request failed: ${errorMessage}`, this.providerName, error);
    }
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *generateStream(input, options = {}) {
    let messagesForChat;
    if (typeof input === 'string') {
      messagesForChat = [{ role: 'user', content: input }];
    } else if (Array.isArray(input)) {
      messagesForChat = input;
    } else {
      throw new LLMPlugRequestError("Invalid input type for generateStream.", this.providerName);
    }
    yield* this.chatStream(messagesForChat, {
        model: options.model,
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        extraParams: { topP: options.extraParams?.topP, randomSeed: options.extraParams?.randomSeed }
    });
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *chatStream(messages, options = {}) {
    const model = options.model || this.defaultModel;
    const mistralMessages = await this._prepareMessages(messages);

    const mistralTools = options.tools?.map(tool => ({
      type: 'function',
      function: {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      }
    }));

    let toolChoiceOption = options.toolChoice;
     if (typeof options.toolChoice !== 'string') {
        toolChoiceOption = 'auto';
    }

    try {
      const stream = this.client.chatStream({
        model: model,
        messages: mistralMessages,
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        topP: options.extraParams?.topP,
        randomSeed: options.extraParams?.randomSeed,
        // safePrompt: options.extraParams?.safePrompt,
        tools: mistralTools,
        toolChoice: toolChoiceOption,
        responseFormat: options.responseFormat?.type === 'json_object' ? { type: 'json_object' } : undefined,
      });

      let currentToolCalls = {}; // Accumulate arguments for tool calls

      for await (const chunk of stream) {
        const chunkData = { rawChunk: chunk };
        const choice = chunk.choices[0];

        if (choice?.delta?.content) {
          chunkData.text = choice.delta.content;
        }

        if (choice?.delta?.tool_calls) {
          const streamedToolCalls = [];
          for (const tcDelta of choice.delta.tool_calls) {
            // tcDelta contains index, id, type, function: { name, arguments }
            // Arguments stream incrementally.
            if (!currentToolCalls[tcDelta.index]) {
              currentToolCalls[tcDelta.index] = {
                id: tcDelta.id, // ID is usually set at the start of the tool call
                type: 'function',
                function: {
                  name: tcDelta.function.name,
                  arguments: tcDelta.function.arguments || ''
                }
              };
            } else {
                // Append to existing arguments if ID and name match (or just rely on index)
                if (tcDelta.function.arguments) {
                    currentToolCalls[tcDelta.index].function.arguments += tcDelta.function.arguments;
                }
                // Update id or name if they appear later (though usually fixed from start)
                if (tcDelta.id) currentToolCalls[tcDelta.index].id = tcDelta.id;
                if (tcDelta.function.name) currentToolCalls[tcDelta.index].function.name = tcDelta.function.name;

            }
            // Yield the current state of the tool call (or only when it's "complete" based on finish_reason)
            // For simplicity, we'll yield the partial tool call with accumulated args.
            streamedToolCalls.push({ ...currentToolCalls[tcDelta.index] });
          }
          if (streamedToolCalls.length > 0) {
            chunkData.toolCalls = streamedToolCalls;
          }
        }
        
        if (choice?.finish_reason) {
          chunkData.finishReason = choice.finish_reason.toLowerCase();
          // Reset currentToolCalls for the next potential set of calls in the same stream (if possible)
          // or typically this signals the end of this turn.
          currentToolCalls = {};
        }

        // Mistral API provides usage stats in the *last* chunk of the stream for some models/endpoints
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
      throw new LLMPlugRequestError(`Mistral AI API chat stream failed: ${errorMessage}`, this.providerName, error);
    }
  }
}