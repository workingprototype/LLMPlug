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
   * @param {string} prompt
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async generate(prompt, options = {}) {
    const model = options.model || this.defaultModel;
    try {
      const completion = await this.client.chat.completions.create({
        model: model,
        messages: [{ role: 'user', content: prompt }],
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stop: options.stopSequences,
        ...options.extraParams,
      });
      return completion.choices[0]?.message?.content?.trim() || '';
    } catch (error) {
      throw new LLMPlugRequestError(`OpenAI API request failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async chat(messages, options = {}) {
    const model = options.model || this.defaultModel;
    try {
      const completion = await this.client.chat.completions.create({
        model: model,
        messages: messages,
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stop: options.stopSequences,
        ...options.extraParams,
      });
      return completion.choices[0]?.message?.content?.trim() || '';
    } catch (error) {
      throw new LLMPlugRequestError(`OpenAI API chat request failed: ${error.message}`, this.providerName, error);
    }
  }
}