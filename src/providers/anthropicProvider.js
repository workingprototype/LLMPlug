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
    this.defaultModel = config.defaultModel || 'claude-2.1'; // Or 'claude-3-opus-20240229', 'claude-3-sonnet...' etc.
  }

  /**
   * Transforms messages for Anthropic: combines consecutive user/assistant messages and ensures user message is last before API call if necessary.
   * Anthropic API expects `system` prompt separately and an alternating sequence of `user` and `assistant` messages.
   * The `messages.create` API has slightly different expectations for the first message if a system prompt is also used.
   * It also doesn't want an empty `messages` array.
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @param {import('./baseProvider.js').GenerationOptions} options
   * @returns {{ systemPrompt?: string, anthropicMessages: Anthropic.Messages.MessageParam[] }}
   */
  _prepareMessages(messages, options) {
    let systemPrompt;
    const anthropicMessages = [];

    messages.forEach(msg => {
      if (msg.role === 'system') {
        systemPrompt = msg.content; // Anthropic takes only one system prompt
      } else if (msg.role === 'user' || msg.role === 'assistant') {
        anthropicMessages.push({ role: msg.role, content: msg.content });
      }
    });
    
    // Ensure there's at least one message for the API if no system prompt
    if (!systemPrompt && anthropicMessages.length === 0 && options.promptForEmpty) {
        anthropicMessages.push({ role: 'user', content: options.promptForEmpty });
    } else if (anthropicMessages.length === 0 && !systemPrompt) {
        throw new LLMPlugRequestError("Anthropic requires at least one user message or a system prompt.", this.providerName);
    }


    return { systemPrompt, anthropicMessages };
  }


  /**
   * @param {string} prompt
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async generate(prompt, options = {}) {
    // Anthropic's primary interface is chat-like, so we adapt.
    // We can use a system prompt for instructions and the user prompt as the first message.
    // Or, if the model supports it well, just a single user message.
    // For claude-3, the `messages` API is preferred.
    const messages = [{ role: 'user', content: prompt }];
    return this.chat(messages, options);
  }

  /**
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async chat(messages, options = {}) {
    const model = options.model || this.defaultModel;
    const { systemPrompt, anthropicMessages } = this._prepareMessages(messages, {promptForEmpty: "Hello."});

    if (anthropicMessages.length === 0 && !systemPrompt) {
        throw new LLMPlugRequestError("Anthropic API requires messages to proceed.", this.providerName);
    }

    try {
      const response = await this.client.messages.create({
        model: model,
        system: systemPrompt, // Optional, only if a system message was provided
        messages: anthropicMessages,
        max_tokens: options.maxTokens || 1024, // Anthropic requires max_tokens
        temperature: options.temperature,
        stop_sequences: options.stopSequences,
        ...options.extraParams,
      });
      // The response content is an array of blocks, usually one text block
      return response.content.map(block => block.type === 'text' ? block.text : '').join('').trim();
    } catch (error) {
      const errorMessage = error.error?.message || error.message || "Unknown Anthropic API error";
      throw new LLMPlugRequestError(`Anthropic API chat request failed: ${errorMessage}`, this.providerName, error);
    }
  }
}