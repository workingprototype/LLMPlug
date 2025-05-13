import { GenericOpenAICompatibleProvider } from './genericOpenAICompatibleProvider.js';

const LLAMACPP_DEFAULT_BASE_URL = "http://localhost:8080/v1"; // Common default for llama.cpp server

export class LlamaCppServerProvider extends GenericOpenAICompatibleProvider {
  constructor(config = {}) {
    super(
      config, 
      "LlamaCppServer", 
      config.baseURL || LLAMACPP_DEFAULT_BASE_URL
    );
    // Llama.cpp server often needs model to be specified as part of the endpoint or as a fixed setting.
    // The 'model' parameter passed to the OpenAI SDK might be treated as an alias or ignored if the server
    // is configured to serve a single model. If it serves multiple, then 'model' should map to the loaded model alias.
    if (!this.defaultModel) {
        console.warn(`[${this.providerName}] Ensure the 'defaultModel' or per-call 'model' name matches a model alias/file loaded by your llama.cpp server.`);
    }
  }

  // Inherits all methods from GenericOpenAICompatibleProvider
  // Specific overrides can be added here if llama.cpp server has unique OpenAI API quirks.
}