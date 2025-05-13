import { LLMPlugError } from '../utils/errors.js';

/**
 * @typedef {Object} ChatMessage
 * @property {'system' | 'user' | 'assistant'} role
 * @property {string} content
 */

/**
 * @typedef {Object} GenerationOptions
 * @property {string} [model] - The specific model to use.
 * @property {number} [temperature] - Sampling temperature.
 * @property {number} [maxTokens] - Maximum number of tokens to generate.
 * @property {string[]} [stopSequences] - Sequences where the API will stop generating.
 * @property {any} [extraParams] - Any other provider-specific parameters.
 */

export class BaseProvider {
  constructor(config) {
    this.config = config;
    this.providerName = "BaseProvider"; // Should be overridden by subclasses
  }

  /**
   * Generates a text completion based on a prompt.
   * @param {string} prompt - The input prompt.
   * @param {GenerationOptions} [options={}] - Options for generation.
   * @returns {Promise<string>} The generated text.
   * @throws {LLMPlugError} If the method is not implemented or an API error occurs.
   */
  async generate(prompt, options = {}) {
    throw new LLMPlugError(`'generate' method not implemented for ${this.providerName}`, this.providerName);
  }

  /**
   * Generates a chat completion based on a series of messages.
   * @param {ChatMessage[]} messages - An array of chat messages.
   * @param {GenerationOptions} [options={}] - Options for generation.
   * @returns {Promise<string>} The assistant's reply.
   * @throws {LLMPlugError} If the method is not implemented or an API error occurs.
   */
  async chat(messages, options = {}) {
    throw new LLMPlugError(`'chat' method not implemented for ${this.providerName}`, this.providerName);
  }

  /**
   * A common method to get the API key, prioritizing direct config over environment variables.
   * @param {string} envVarName - The environment variable name for the API key.
   * @param {string} [configKeyName='apiKey'] - The key name in the constructor's config object.
   * @returns {string} The API key.
   * @throws {LLMPlugConfigurationError} If the API key is not found.
   */
  _getApiKey(envVarName, configKeyName = 'apiKey') {
    const key = this.config[configKeyName] || process.env[envVarName];
    if (!key) {
      throw new LLMPlugConfigurationError(
        `API key not found. Set it in config.${configKeyName} or as ${envVarName} environment variable.`,
        this.providerName
      );
    }
    return key;
  }
}