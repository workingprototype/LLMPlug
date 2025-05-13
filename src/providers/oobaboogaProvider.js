import { GenericOpenAICompatibleProvider } from './genericOpenAICompatibleProvider.js';

const OOBABOOGA_DEFAULT_BASE_URL = "http://localhost:5000/v1"; // Common default for Oobabooga OpenAI extension

export class OobaboogaProvider extends GenericOpenAICompatibleProvider {
  constructor(config = {}) {
    super(
      config, 
      "Oobabooga", 
      config.baseURL || OOBABOOGA_DEFAULT_BASE_URL
    );
    // Oobabooga's OpenAI extension usually serves the currently loaded model.
    // The 'model' parameter in the request is often used by the extension to ensure
    // it matches, or it might be ignored if only one model is active.
    if (!this.defaultModel) {
        console.warn(`[${this.providerName}] 'defaultModel' or per-call 'model' should ideally match the model loaded in Oobabooga Text Generation WebUI. It might be ignored by the server if only one model is active.`);
    }
  }

  // Inherits all methods from GenericOpenAICompatibleProvider
  // Specific overrides can be added here if Oobabooga's API has unique OpenAI API quirks.
}