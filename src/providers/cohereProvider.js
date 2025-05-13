import { CohereClient, CohereError } from 'cohere-ai';
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError, LLMPlugToolError } from '../utils/errors.js';

export class CohereProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "Cohere";
    try {
      this.apiKey = this._getApiKey('COHERE_API_KEY');
      this.client = new CohereClient({ token: this.apiKey });
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error;
      throw new LLMPlugConfigurationError(`Cohere client initialization failed: ${error.message}`, this.providerName, error);
    }
    this.defaultChatModel = config.defaultChatModel || 'command-r'; // Good for chat and tool use
    this.defaultGenerateModel = config.defaultGenerateModel || 'command-light'; // Good for generation tasks
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async generate(input, options = {}) {
    const model = options.model || this.defaultGenerateModel;
    let promptText;

    if (typeof input === 'string') {
      promptText = input;
    } else if (Array.isArray(input)) {
      // Convert chat messages to a single prompt string for Cohere's generate endpoint
      promptText = input.map(msg => {
        const contentText = Array.isArray(msg.content) ? msg.content.filter(p => p.type === 'text').map(p => p.text).join('\n') : msg.content;
        return `${msg.role}: ${contentText}`;
      }).join('\n\n');
    } else {
      throw new LLMPlugRequestError("Invalid input type for generate. Must be string or ChatMessage[].", this.providerName);
    }

    if (!promptText) {
        throw new LLMPlugRequestError("Prompt text cannot be empty for Cohere generate.", this.providerName);
    }

    try {
      const response = await this.client.generate({
        prompt: promptText,
        model: model,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
        k: options.extraParams?.k, // Top-k sampling
        p: options.extraParams?.p, // Top-p (nucleus) sampling
        stopSequences: options.stopSequences,
        returnLikelihoods: options.extraParams?.returnLikelihoods, // 'GENERATION', 'ALL', or 'NONE'
        // Cohere's generate doesn't directly support JSON mode or tool use
      });

      const generation = response.generations?.[0];
      if (!generation) {
        throw new LLMPlugRequestError("Cohere API returned no generations.", this.providerName, response);
      }

      // Cohere's generate endpoint doesn't typically provide token counts in the same way chat does.
      // We can get it if `returnLikelihoods` is not NONE, but it's more complex.
      // For simplicity, we'll mark usage as null for generate.
      return {
        text: generation.text.trim(),
        usage: null, // Detailed token usage not directly available from generate like chat
        finishReason: generation.finishReason?.toLowerCase(), // e.g. COMPLETE, MAX_TOKENS, ERROR_TOXIC
        rawResponse: response,
      };
    } catch (error) {
        if (error instanceof CohereError) {
            throw new LLMPlugRequestError(`Cohere API generate request failed: ${error.message} (Status: ${error.statusCode})`, this.providerName, error);
        }
        throw new LLMPlugRequestError(`Cohere API generate request failed: ${error.message}`, this.providerName, error);
    }
  }

  _formatMessagesForCohere(messages) {
    // Cohere's chat expects: { role: "USER" | "CHATBOT" | "SYSTEM" | "TOOL", message: string }
    // Tool results: { role: "TOOL", tool_results: [{ call: {...}, outputs: [{...}]}] }
    const cohereMessages = [];
    let systemMessage;

    for (const msg of messages) {
        const roleMap = {
            user: 'USER',
            assistant: 'CHATBOT',
            system: 'SYSTEM',
            tool: 'TOOL',
        };
        const cohereRole = roleMap[msg.role];
        if (!cohereRole) {
            console.warn(`CohereProvider: Unsupported role '${msg.role}' will be mapped to USER.`);
            cohereRole = 'USER';
        }

        if (cohereRole === 'SYSTEM') {
            systemMessage = typeof msg.content === 'string' ? msg.content : msg.content.map(c => c.type === 'text' ? c.text : '').join('\n');
            continue; // System message is handled separately in the chat request
        }
        
        if (cohereRole === 'TOOL') {
            if (!msg.tool_call_id || !Array.isArray(msg.content)) {
                 throw new LLMPlugToolError("Cohere: 'tool' role message must have tool_call_id and array content with 'tool_output'.", this.providerName, msg.name);
            }
            const toolOutputPart = msg.content.find(p => p.type === 'tool_output');
            if (!toolOutputPart) {
                 throw new LLMPlugToolError("Cohere: 'tool' role message content must contain a 'tool_output' part.", this.providerName, msg.name);
            }

            // Find the original tool call that this result corresponds to.
            // This requires looking back in the `messages` array for the assistant's tool_calls.
            // This is a bit complex as Cohere's tool_results structure expects the `call` object.
            // For simplicity, we'll assume the necessary `call` object can be reconstructed or is passed via `extraParams`.
            // A more robust solution would involve carrying the original `ToolCall` object from the assistant's turn.

            // Simplified: assuming the user is responsible for structuring tool_results correctly for now,
            // or we find the original call from history.
            // The Cohere SDK expects tool_results to be an array of objects, where each object contains
            // the `call` (the original tool_call from the model) and `outputs` (an array of outputs for that call).
             const originalToolCallFromHistory = messages
                .filter(m => m.role === 'assistant' && m.tool_calls)
                .flatMap(m => m.tool_calls)
                .find(tc => tc.id === msg.tool_call_id);

            if (!originalToolCallFromHistory) {
                // If we can't find it, we can't form the 'call' object Cohere needs
                // This highlights a need for more robust state management or passing of tool call details
                // For now, we'll create a placeholder or throw
                throw new LLMPlugToolError(`Cohere: Could not find original tool call for ID ${msg.tool_call_id} to construct TOOL message. Original call details are needed.`, this.providerName, msg.name);
            }

            cohereMessages.push({
                role: 'TOOL',
                toolResults: [{
                    call: { // Reconstruct or use stored original tool call details
                        name: originalToolCallFromHistory.function.name,
                        // Cohere expects parameters as a map, not a JSON string
                        parameters: JSON.parse(originalToolCallFromHistory.function.arguments || '{}')
                    },
                    // Cohere tool outputs are an array of objects (dictionaries)
                    outputs: [ typeof toolOutputPart.content === 'object' ? toolOutputPart.content : { result: toolOutputPart.content } ]
                }]
            });

        } else { // USER or CHATBOT
            let messageText = "";
            if (typeof msg.content === 'string') {
                messageText = msg.content;
            } else if (Array.isArray(msg.content)) {
                // Cohere chat does not support direct image URLs in the same way as multimodal models.
                // It primarily expects text. We will concatenate text parts.
                messageText = msg.content.filter(p => p.type === 'text').map(p => p.text).join('\n');
                if (msg.content.some(p => p.type === 'image_url')) {
                    console.warn("CohereProvider: Image content parts are ignored for chat messages as Cohere chat API is primarily text-based.");
                }
            }

            if (cohereRole === 'CHATBOT' && msg.tool_calls && msg.tool_calls.length > 0) {
                // Assistant message includes tool calls
                cohereMessages.push({
                    role: 'CHATBOT',
                    message: messageText || " ", // Message can be empty if only making tool calls
                    toolCalls: msg.tool_calls.map(tc => ({
                        name: tc.function.name,
                        parameters: JSON.parse(tc.function.arguments || '{}') // Cohere expects parsed parameters
                    }))
                });
            } else if (messageText) {
                 cohereMessages.push({ role: cohereRole, message: messageText });
            } else if (cohereRole === 'CHATBOT' && !messageText && (!msg.tool_calls || msg.tool_calls.length === 0)) {
                // Empty assistant message
                cohereMessages.push({ role: 'CHATBOT', message: " " }); // Send a space if completely empty
            }
        }
    }
    return { preamble: systemMessage, messages: cohereMessages };
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<import('../baseProvider.js').GenerationResult>}
   */
  async chat(messages, options = {}) {
    const model = options.model || this.defaultChatModel;
    const { preamble, messages: chatHistory } = this._formatMessagesForCohere(messages);

    // The last message in our formatted `chatHistory` is usually the user's current query.
    // Cohere's API takes `message` (current user query) and `chatHistory` (previous turns).
    let currentMessage = "";
    let cohereChatHistory = [];

    if (chatHistory.length > 0) {
        const lastFormattedMsg = chatHistory[chatHistory.length - 1];
        // Check if the last message is a USER message to treat as current query
        if (lastFormattedMsg.role === 'USER') {
            currentMessage = lastFormattedMsg.message;
            cohereChatHistory = chatHistory.slice(0, -1);
        } else {
            // If last message is not USER (e.g. CHATBOT or TOOL), then there's no new "message" for Cohere,
            // and the entire chatHistory is history. This implies we expect the model to continue.
            // This scenario is less common for a typical "user asks, bot replies" flow.
            // Forcing a tool call response might fall here.
            cohereChatHistory = chatHistory;
            // If currentMessage is empty, Cohere might complain unless it's a tool result flow.
            // This might require a dummy " " message if the SDK is strict.
            if (cohereChatHistory.length > 0 && cohereChatHistory[cohereChatHistory.length -1].role === 'TOOL') {
                currentMessage = "Okay, continue based on the tool results."; // Or some other neutral continuation
            } else if (cohereChatHistory.length === 0 && !preamble) {
                throw new LLMPlugRequestError("Cohere chat: No current message or history to send.", this.providerName);
            }
        }
    } else if (!preamble) {
         throw new LLMPlugRequestError("Cohere chat: No messages or preamble to send.", this.providerName);
    }


    const cohereTools = options.tools?.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        parameterDefinitions: tool.function.parameters?.properties ? 
            Object.fromEntries(Object.entries(tool.function.parameters.properties).map(([key, value]) => [
                key,
                { description: value.description, type: value.type.toLowerCase(), required: tool.function.parameters.required?.includes(key) || false }
            ])) : undefined,
    }));

    const request = {
      message: currentMessage, // Can be empty if chatHistory has tool results to process
      chatHistory: cohereChatHistory,
      preamble: preamble,
      model: model,
      temperature: options.temperature,
      maxTokens: options.maxTokens, // Max output tokens
      k: options.extraParams?.k,
      p: options.extraParams?.p,
      stopSequences: options.stopSequences,
      // promptTruncation: 'AUTO', // Default
      tools: cohereTools,
      // toolResults: if the last turn was CHATBOT making tool_calls, and this turn is providing results.
      // This is handled by formatting TOOL messages in _formatMessagesForCohere.
      // forceSingleStep: false, // If true, model won't make tool calls.
      // Cohere chat doesn't have a direct JSON mode flag like OpenAI.
      // You'd instruct it via prompt if you want JSON.
      ...options.extraParams,
    };
    
    // If currentMessage is genuinely empty and not a tool flow, some models might require a non-empty message.
    if (!request.message && request.chatHistory.every(m => m.role !== 'TOOL')) {
        request.message = " "; // Send a space if it's not a tool flow continuation
    }


    try {
      const response = await this.client.chat(request);

      const textContent = response.text?.trim() || null;
      const toolCalls = response.toolCalls?.map(tc => ({
        id: `${tc.name}-${Date.now()}`, // Cohere tool calls don't have IDs, generate one
        type: 'function',
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.parameters || {}),
        },
      })) || [];

      // Cohere's chat response provides detailed token counts via `meta`
      const usage = {
        promptTokens: response.meta?.tokens?.inputTokens,
        completionTokens: response.meta?.tokens?.outputTokens,
        totalTokens: response.meta?.tokens?.inputTokens + response.meta?.tokens?.outputTokens,
      };
      
      // Cohere's finishReason in chat: "COMPLETE", "MAX_TOKENS", "ERROR", "ERROR_TOXIC", "ERROR_LIMIT"
      // Also "TOOL_CALLS" if it made tool calls.
      const finishReason = response.finishReason?.toUpperCase();

      return {
        text: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: usage,
        finishReason: finishReason,
        rawResponse: response,
      };
    } catch (error) {
        if (error instanceof CohereError) {
            throw new LLMPlugRequestError(`Cohere API chat request failed: ${error.message} (Status: ${error.statusCode}) Body: ${JSON.stringify(error.body)}`, this.providerName, error);
        }
      throw new LLMPlugRequestError(`Cohere API chat request failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {string | import('../baseProvider.js').ChatMessage[]} input
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *generateStream(input, options = {}) {
    // Cohere's generate endpoint also supports streaming
    const model = options.model || this.defaultGenerateModel;
    let promptText;

    if (typeof input === 'string') {
      promptText = input;
    } else if (Array.isArray(input)) {
      promptText = input.map(msg => `${msg.role}: ${Array.isArray(msg.content) ? msg.content.filter(p=>p.type==='text').map(p=>p.text).join('\n') : msg.content}`).join('\n\n');
    } else {
      throw new LLMPlugRequestError("Invalid input type for generateStream.", this.providerName);
    }

    if (!promptText) {
        throw new LLMPlugRequestError("Prompt text cannot be empty for Cohere generateStream.", this.providerName);
    }

    try {
      const stream = await this.client.generateStream({
        prompt: promptText,
        model: model,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
        k: options.extraParams?.k,
        p: options.extraParams?.p,
        stopSequences: options.stopSequences,
      });

      for await (const event of stream) {
        const chunkData = { rawChunk: event };
        if (event.eventType === 'text-generation' && event.text) {
          chunkData.text = event.text;
        } else if (event.eventType === 'stream-end') {
          chunkData.finishReason = event.finishReason?.toLowerCase();
          // Usage data not typically provided per chunk or at end of generateStream in same detail as chatStream
        }
        yield chunkData;
      }
    } catch (error) {
        if (error instanceof CohereError) {
             throw new LLMPlugRequestError(`Cohere API generateStream failed: ${error.message} (Status: ${error.statusCode})`, this.providerName, error);
        }
      throw new LLMPlugRequestError(`Cohere API generateStream failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {import('../baseProvider.js').ChatMessage[]} messages
   * @param {import('../baseProvider.js').GenerationOptions} [options={}]
   * @returns {AsyncIterable<import('../baseProvider.js').GenerationStreamChunk>}
   */
  async *chatStream(messages, options = {}) {
    const model = options.model || this.defaultChatModel;
    const { preamble, messages: chatHistory } = this._formatMessagesForCohere(messages);

    let currentMessage = "";
    let cohereChatHistory = [];
    if (chatHistory.length > 0) {
        const lastFormattedMsg = chatHistory[chatHistory.length - 1];
        if (lastFormattedMsg.role === 'USER') {
            currentMessage = lastFormattedMsg.message;
            cohereChatHistory = chatHistory.slice(0, -1);
        } else {
            cohereChatHistory = chatHistory;
             if (cohereChatHistory.length > 0 && cohereChatHistory[cohereChatHistory.length -1].role === 'TOOL') {
                currentMessage = "Okay, continue based on the tool results.";
            }
        }
    } else if (!preamble) {
         throw new LLMPlugRequestError("Cohere chatStream: No messages or preamble to send.", this.providerName);
    }
    
    if (!currentMessage && cohereChatHistory.every(m => m.role !== 'TOOL')) {
        currentMessage = " "; 
    }

    const cohereTools = options.tools?.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        parameterDefinitions: tool.function.parameters?.properties ? 
            Object.fromEntries(Object.entries(tool.function.parameters.properties).map(([key, value]) => [
                key,
                { description: value.description, type: value.type.toLowerCase(), required: tool.function.parameters.required?.includes(key) || false }
            ])) : undefined,
    }));

    const request = {
      message: currentMessage,
      chatHistory: cohereChatHistory,
      preamble: preamble,
      model: model,
      temperature: options.temperature,
      // maxTokens for chatStream is often interpreted as total tokens for the turn, not just output.
      // Cohere's API for stream might not use maxTokens in the same way as non-streaming.
      // It typically streams until a stop condition or model decides to end.
      // We can pass it, but its effect might vary.
      maxTokens: options.maxTokens,
      k: options.extraParams?.k,
      p: options.extraParams?.p,
      stopSequences: options.stopSequences,
      tools: cohereTools,
      ...options.extraParams,
    };
    
    try {
      const stream = await this.client.chatStream(request);
      let accumulatedToolCallParams = {}; // { [toolName]: "partial_json_args" }

      for await (const event of stream) {
        const chunkData = { rawChunk: event };

        if (event.eventType === 'text-generation' && event.text) {
          chunkData.text = event.text;
        } else if (event.eventType === 'tool-calls-generation') {
            // This event type provides the names and initial (often empty) parameters of tools the model wants to call.
            // The actual parameters stream via 'tool-calls-chunk'.
            // The Cohere SDK might already aggregate this, but we can prepare.
            if (event.toolCalls && event.toolCalls.length > 0) {
                chunkData.toolCalls = event.toolCalls.map(tc => {
                    accumulatedToolCallParams[tc.name] = ''; // Initialize accumulator
                    return {
                        id: `${tc.name}-stream-${Date.now()}`, // Generate ID
                        type: 'function',
                        function: { name: tc.name, arguments: '' } // Arguments will be filled
                    };
                });
            }
        } else if (event.eventType === 'tool-calls-chunk') {
            // This is where Cohere streams the parameters for the tool calls.
            // The 'text' field in this event contains a chunk of the JSON parameters.
            // We need to associate this with the correct tool call.
            // This part is tricky as the event doesn't directly link to a specific toolCall from the 'tool-calls-generation' event.
            // Assuming it relates to the most recently announced tool call or a single tool call if only one.
            // A more robust implementation would need careful state management based on Cohere's streaming specifics.
            // For now, if there's one active tool call being streamed, append to it.
            const activeToolNames = Object.keys(accumulatedToolCallParams);
            if (activeToolNames.length === 1 && event.text) { // Simplistic: assumes one tool call's params stream at a time
                const toolName = activeToolNames[0];
                accumulatedToolCallParams[toolName] += event.text;
                // We don't yield toolCalls on every chunk of arguments, but at the end or when a new tool call starts.
            }

        } else if (event.eventType === 'stream-end') {
          chunkData.finishReason = event.finishReason?.toUpperCase();
          if (event.response) { // Final response object at stream end
            if (event.response.meta?.tokens) {
              chunkData.usage = {
                promptTokens: event.response.meta.tokens.inputTokens,
                completionTokens: event.response.meta.tokens.outputTokens,
                totalTokens: event.response.meta.tokens.inputTokens + event.response.meta.tokens.outputTokens,
              };
            }
            // Finalize any accumulated tool calls if params were streamed
            const finalToolCalls = [];
            for (const toolName in accumulatedToolCallParams) {
                finalToolCalls.push({
                    id: `${toolName}-stream-final-${Date.now()}`,
                    type: 'function',
                    function: { name: toolName, arguments: accumulatedToolCallParams[toolName] }
                });
            }
            if (finalToolCalls.length > 0) {
                chunkData.toolCalls = finalToolCalls;
            } else if (event.response.toolCalls && event.response.toolCalls.length > 0) {
                // If tool calls were not streamed by param chunks but came fully formed at the end
                 chunkData.toolCalls = event.response.toolCalls.map(tc => ({
                    id: `${tc.name}-stream-end-${Date.now()}`,
                    type: 'function',
                    function: { name: tc.name, arguments: JSON.stringify(tc.parameters || {}) }
                }));
            }
          }
        }
        yield chunkData;
      }
    } catch (error) {
        if (error instanceof CohereError) {
            throw new LLMPlugRequestError(`Cohere API chatStream failed: ${error.message} (Status: ${error.statusCode}) Body: ${JSON.stringify(error.body)}`, this.providerName, error);
        }
      throw new LLMPlugRequestError(`Cohere API chatStream failed: ${error.message}`, this.providerName, error);
    }
  }
}