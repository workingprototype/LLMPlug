import { OpenAIProvider } from './providers/openaiProvider.js';
import { AnthropicProvider } from './providers/anthropicProvider.js';
import { GoogleProvider } from './providers/googleProvider.js';
import { HuggingFaceProvider } from './providers/huggingfaceProvider.js';
import { LLMPlugError } from './utils/errors.js';

const PROVIDERS = {
  openai: OpenAIProvider,
  anthropic: AnthropicProvider,
  google: GoogleProvider,
  huggingface: HuggingFaceProvider,
  // TODO: Add more providers here as they are implemented
};

export class LLMPlug {
  /**
   * Gets an instance of a specific LLM provider.
   * @param {keyof PROVIDERS} providerName - The name of the provider (e.g., 'openai', 'anthropic').
   * @param {object} [config={}] - Provider-specific configuration.
   *                               For OpenAI: { apiKey?, defaultModel? }
   *                               For Anthropic: { apiKey?, defaultModel? }
   *                               For Google: { apiKey?, defaultModel? }
   *                               For HuggingFace: { apiToken?, modelId!, task? }
   * @returns {import('./providers/baseProvider.js').BaseProvider} An instance of the requested provider.
   * @throws {LLMPlugError} If the provider is not supported.
   */
  static getProvider(providerName, config = {}) {
    const ProviderClass = PROVIDERS[providerName.toLowerCase()];
    if (!ProviderClass) {
      throw new LLMPlugError(`Unsupported provider: ${providerName}. Supported providers are: ${Object.keys(PROVIDERS).join(', ')}`);
    }
    return new ProviderClass(config);
  }
}

// Export individual providers for direct use if desired, though getProvider is recommended
export { OpenAIProvider, AnthropicProvider, GoogleProvider, HuggingFaceProvider };
export * from './utils/errors.js'; // Export error classes