import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

export class GoogleProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "Google";
    try {
      this.apiKey = this._getApiKey('GOOGLE_GEMINI_API_KEY');
      this.genAI = new GoogleGenerativeAI(this.apiKey);
    } catch (error) {
      if (error instanceof LLMPlugConfigurationError) throw error;
      throw new LLMPlugConfigurationError(`Google AI client initialization failed: ${error.message}`, this.providerName, error);
    }
    this.defaultModel = config.defaultModel || 'gemini-2.0-pro-exp-02-05'; // Or 'gemini-1.5-flash', 'gemini-1.5-pro' etc.
    
    // Default safety settings (can be overridden via options.extraParams.safetySettings)
    this.defaultSafetySettings = [
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    ];
  }

  /**
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @returns {Object[]} Google AI specific message format
   */
  _formatMessagesForGoogle(messages) {
    // Gemini expects 'parts' and alternates 'user' and 'model' roles.
    // System instructions can be handled differently or at the start.
    // For simplicity, we'll treat 'system' as the first 'user' message if present.
    const history = [];
    let currentRole = 'user'; // API expects alternating roles, starting with user
    let systemMessageContent = "";

    messages.forEach(msg => {
      if (msg.role === 'system') {
        systemMessageContent += (systemMessageContent ? "\n" : "") + msg.content;
      } else if (msg.role === 'user') {
        if (systemMessageContent) { // Prepend system content to first user message
            history.push({ role: 'user', parts: [{ text: systemMessageContent + "\n" + msg.content }] });
            systemMessageContent = ""; // Clear after use
        } else {
            history.push({ role: 'user', parts: [{ text: msg.content }] });
        }
        currentRole = 'model';
      } else if (msg.role === 'assistant') {
        // If the last message was also 'assistant' (model), this is an issue for Gemini's strict alternation.
        // This simple library won't handle merging. For now, we assume valid alternation from input.
        history.push({ role: 'model', parts: [{ text: msg.content }] });
        currentRole = 'user';
      }
    });
    
    // If there's remaining system content and no user messages, treat it as a user prompt
    if (systemMessageContent && history.length === 0) {
        history.push({ role: 'user', parts: [{ text: systemMessageContent }] });
    }
    
    return history;
  }

  /**
   * @param {string} prompt
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async generate(prompt, options = {}) {
    const modelName = options.model || this.defaultModel;
    const model = this.genAI.getGenerativeModel({ 
      model: modelName,
      safetySettings: options.extraParams?.safetySettings || this.defaultSafetySettings,
    });

    try {
      const generationConfig = {
        temperature: options.temperature,
        maxOutputTokens: options.maxTokens,
        stopSequences: options.stopSequences,
        ...(options.extraParams?.generationConfig || {}),
      };

      const result = await model.generateContent(prompt, generationConfig);
      const response = result.response;
      return response.text();
    } catch (error) {
      throw new LLMPlugRequestError(`Google AI API request failed: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async chat(messages, options = {}) {
    const modelName = options.model || this.defaultModel;
    const model = this.genAI.getGenerativeModel({ 
      model: modelName,
      safetySettings: options.extraParams?.safetySettings || this.defaultSafetySettings,
    });
    
    const formattedMessages = this._formatMessagesForGoogle(messages);
    
    // The last message in formattedMessages is the current prompt to the model.
    // The preceding messages are the history.
    if (formattedMessages.length === 0) {
        throw new LLMPlugRequestError("Google AI chat requires at least one message.", this.providerName);
    }

    const currentPromptObject = formattedMessages.pop(); // Last message is the new prompt
    const history = formattedMessages; // The rest is history

    try {
      const chatSession = model.startChat({
        history: history,
        generationConfig: {
          temperature: options.temperature,
          maxOutputTokens: options.maxTokens,
          stopSequences: options.stopSequences,
          ...(options.extraParams?.generationConfig || {}),
        }
      });

      const result = await chatSession.sendMessage(currentPromptObject.parts.map(p => p.text).join("\n"));
      const response = result.response;
      return response.text();
    } catch (error) {
      throw new LLMPlugRequestError(`Google AI API chat request failed: ${error.message}`, this.providerName, error);
    }
  }
}