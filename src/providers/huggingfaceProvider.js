import fetch from 'node-fetch'; // Make sure 'node-fetch' is in package.json dependencies
import { BaseProvider } from './baseProvider.js';
import { LLMPlugConfigurationError, LLMPlugRequestError } from '../utils/errors.js';

const HUGGINGFACE_API_BASE_URL = "https://api-inference.huggingface.co/models/";

export class HuggingFaceProvider extends BaseProvider {
  constructor(config = {}) {
    super(config);
    this.providerName = "HuggingFace";
    try {
      // API token is required for non-public models or to avoid rate limits
      this.apiToken = this._getApiKey('HUGGINGFACE_API_TOKEN', 'apiToken'); // Allow passing token directly
    } catch (error) {
      // Token is optional for public models, so don't throw if not found, but warn.
      console.warn("HuggingFace API token not found. For private models or higher rate limits, please provide one.");
      this.apiToken = null;
    }
    // User MUST provide a model_id for Hugging Face as there's no sensible default
    if (!config.modelId) {
      throw new LLMPlugConfigurationError("`modelId` is required in config for HuggingFaceProvider (e.g., 'gpt2', 'mistralai/Mistral-7B-Instruct-v0.1').", this.providerName);
    }
    this.modelId = config.modelId;
    this.task = config.task || 'text-generation'; // or 'conversational'
  }

  async _makeApiCall(payload, modelIdOverride = null, taskOverride = null) {
    const effectiveModelId = modelIdOverride || this.modelId;
    const effectiveTask = taskOverride || this.task;
    const apiUrl = `${HUGGINGFACE_API_BASE_URL}${effectiveModelId}`;

    const headers = {
      'Content-Type': 'application/json',
    };
    if (this.apiToken) {
      headers['Authorization'] = `Bearer ${this.apiToken}`;
    }

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new LLMPlugRequestError(
          `Hugging Face API request failed for model ${effectiveModelId} with status ${response.status}: ${errorBody}`,
          this.providerName
        );
      }
      return await response.json();
    } catch (error) {
      if (error instanceof LLMPlugRequestError) throw error;
      throw new LLMPlugRequestError(`Hugging Face API request error: ${error.message}`, this.providerName, error);
    }
  }

  /**
   * @param {string} prompt
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async generate(prompt, options = {}) {
    const modelId = options.model || this.modelId; // Allow overriding model per call
    const payload = {
      inputs: prompt,
      parameters: {
        max_new_tokens: options.maxTokens, // HF uses max_new_tokens
        temperature: options.temperature,
        return_full_text: false, // Typically you want only the generated part
        stop_sequences: options.stopSequences,
        ...(options.extraParams || {}),
      },
      options: { // For things like wait_for_model
        wait_for_model: true, // Wait if model is loading
        use_cache: options.extraParams?.use_cache !== undefined ? options.extraParams.use_cache : true,
      }
    };

    const apiResponse = await this._makeApiCall(payload, modelId, 'text-generation');
    
    // Response format can vary slightly. Common is an array with one object.
    if (Array.isArray(apiResponse) && apiResponse.length > 0 && apiResponse[0].generated_text) {
      return apiResponse[0].generated_text.trim();
    } else if (apiResponse.generated_text) { // Some models might return object directly
        return apiResponse.generated_text.trim();
    }
    // Handle other potential formats or log warning
    console.warn("HuggingFace generate: Unexpected response format", apiResponse);
    return '';
  }

  /**
   * @param {import('./baseProvider.js').ChatMessage[]} messages
   * @param {import('./baseProvider.js').GenerationOptions} [options={}]
   * @returns {Promise<string>}
   */
  async chat(messages, options = {}) {
    const modelId = options.model || this.modelId;
    const pastUserInputs = [];
    const generatedResponses = [];
    let currentQuery = "";

    messages.forEach(msg => {
      if (msg.role === 'user') {
        if (currentQuery) { // If previous was also user (shouldn't happen in ideal alternating chat)
          pastUserInputs.push(currentQuery); // Treat previous as done
          generatedResponses.push(""); // Add an empty assistant response for it
        }
        currentQuery = msg.content;
      } else if (msg.role === 'assistant') {
        pastUserInputs.push(currentQuery); // The user input that led to this assistant response
        generatedResponses.push(msg.content);
        currentQuery = ""; // Reset current query
      } else if (msg.role === 'system') {
        // Prepend system prompt to the first user query if this model supports it this way
        // Or some models might take it in `parameters`. For simplicity, prepend here.
        currentQuery = msg.content + "\n" + (currentQuery || "");
      }
    });
    
    // The last user message is the one we're sending now.
    // If messages end with assistant, it means we are just providing history.
    // This basic HF client expects to generate a new response.
    if (!currentQuery && pastUserInputs.length > 0) {
        // If the last message was an assistant, we can't directly use conversational endpoint
        // without a new user query. We could try to get the last user input and re-send.
        // For simplicity, let's require the last message to be effectively 'user'.
        // Or, use the last user input from history if available
        if (messages[messages.length-1].role !== 'user' && messages.length > 0) {
            const lastUserMsg = messages.slice().reverse().find(m => m.role === 'user');
            if (lastUserMsg) currentQuery = lastUserMsg.content;
            else throw new LLMPlugRequestError("HuggingFace chat: Last message must be from user, or provide a new user prompt.", this.providerName);
        }
    }


    const payload = {
      inputs: {
        text: currentQuery, // The current user input
        past_user_inputs: pastUserInputs,
        generated_responses: generatedResponses,
      },
      parameters: {
        max_new_tokens: options.maxTokens,
        temperature: options.temperature,
        stop_sequences: options.stopSequences,
        ...(options.extraParams || {}),
      },
      options: {
        wait_for_model: true,
        use_cache: options.extraParams?.use_cache !== undefined ? options.extraParams.use_cache : true,
      }
    };
    
    const apiResponse = await this._makeApiCall(payload, modelId, 'conversational');
    
    if (apiResponse && apiResponse.generated_text) {
      return apiResponse.generated_text.trim();
    }
    // The conversational task might also return `conversation` object with past inputs/outputs
    if (apiResponse && apiResponse.conversation && apiResponse.conversation.generated_responses) {
        const newResponses = apiResponse.conversation.generated_responses;
        if (newResponses.length > generatedResponses.length) {
            return newResponses[newResponses.length -1].trim();
        }
    }

    console.warn("HuggingFace chat: Unexpected response format", apiResponse);
    return '';
  }
}