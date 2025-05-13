import { OpenAIProvider } from './providers/openaiProvider.js';
import { AnthropicProvider } from './providers/anthropicProvider.js';
import { GoogleProvider } from './providers/googleProvider.js';
import { HuggingFaceProvider } from './providers/huggingfaceProvider.js';
import { CohereProvider } from './providers/cohereProvider.js';
import { MistralProvider } from './providers/mistralProvider.js';
import { OpenRouterProvider } from './providers/openRouterProvider.js';
import { OllamaProvider } from './providers/ollamaProvider.js';
import { LlamaCppServerProvider } from './providers/llamaCppServerProvider.js';
import { OobaboogaProvider } from './providers/oobaboogaProvider.js';
import { LLMPlugError } from './utils/errors.js';

const PROVIDERS = {
  openai: OpenAIProvider,
  anthropic: AnthropicProvider,
  google: GoogleProvider,
  huggingface: HuggingFaceProvider,
  cohere: CohereProvider,
  mistralai: MistralProvider,
  openrouter: OpenRouterProvider,
  ollama: OllamaProvider,
  llamacpp: LlamaCppServerProvider,
  oobabooga: OobaboogaProvider,
  // TODO: Add more providers here as they are implemented
};

export class LLMPlug {
  /**
   * Gets an instance of a specific LLM provider.
   * @param {keyof PROVIDERS} providerName - The name of the provider.
   * @param {object} [config={}] - Provider-specific configuration.
   *                               For local providers (Ollama, LlamaCpp, Oobabooga):
   *                                 `baseURL` (e.g., "http://localhost:11434/v1")
   *                                 `defaultModel` (name of the model loaded/served locally)
   *                                 `apiKey` (optional, usually not needed or a dummy string)
   * @returns {import('./providers/baseProvider.js').BaseProvider} An instance of the requested provider.
   * @throws {LLMPlugError} If the provider is not supported.
   */
  static getProvider(providerName, config = {}) {
    const lowerProviderName = providerName.toLowerCase();
    const ProviderClass = PROVIDERS[lowerProviderName];
    
    if (!ProviderClass) {
      throw new LLMPlugError(`Unsupported provider: ${providerName}. Supported providers are: ${Object.keys(PROVIDERS).join(', ')}`);
    }
    return new ProviderClass(config);
  }
}

// Export individual providers for direct use if desired, though getProvider is recommended
export * from './providers/openaiProvider.js';
export * from './providers/anthropicProvider.js';
export * from './providers/googleProvider.js';
export * from './providers/huggingfaceProvider.js';
export * from './providers/cohereProvider.js';
export * from './providers/mistralProvider.js';
export * from './providers/openRouterProvider.js';
export * from './providers/ollamaProvider.js';
export * from './providers/llamaCppServerProvider.js';
export * from './providers/oobaboogaProvider.js';
export * from './providers/genericOpenAICompatibleProvider.js';

export * from './utils/errors.js'; // Export error classes