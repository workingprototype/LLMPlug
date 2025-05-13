import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold, FunctionDeclarationSchemaType } from "@google/generative-ai";
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
    this.defaultModel = config.defaultModel || 'gemini-1.0-pro'; // Changed back to 1.0-pro for simpleUsage
    
    this.defaultSafetySettings = [
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    ];
    // More permissive safety settings for testing code generation (use with caution)
    this.permissiveSafetySettings = [
        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    ];
  }

  async _prepareMessages(messages) {
    let systemInstruction;
    const history = [];

    const systemMsg = messages.find(m => m.role === 'system');
    if (systemMsg) {
      let systemText = '';
      if (typeof systemMsg.content === 'string') {
        systemText = systemMsg.content;
      } else if (Array.isArray(systemMsg.content) && systemMsg.content[0]?.type === 'text') {
        systemText = systemMsg.content.map(c => c.text || '').join('\n');
      }
      if (systemText) {
        // For gemini-1.0-pro, system instructions are best prepended to the first user message.
        // For gemini-1.5-pro, systemInstruction object can be used.
        // We'll adapt based on model or a config flag later if needed. For now, universal approach:
         if (this.defaultModel.startsWith('gemini-1.5')) {
            systemInstruction = { role: 'system', parts: [{ text: systemText }] };
         } else {
            // Prepend to first user message for older models
            const firstUserMsgIndex = messages.findIndex(m => m.role === 'user' && m !== systemMsg);
            if (firstUserMsgIndex !== -1 && messages[firstUserMsgIndex]) {
                let content = messages[firstUserMsgIndex].content;
                if (typeof content === 'string') {
                    messages[firstUserMsgIndex].content = `${systemText}\n\n${content}`;
                } else if (Array.isArray(content)) {
                    const firstTextPartIndex = content.findIndex(p => p.type === 'text');
                    if (firstTextPartIndex !== -1) {
                        content[firstTextPartIndex].text = `${systemText}\n\n${content[firstTextPartIndex].text}`;
                    } else {
                        content.unshift({ type: 'text', text: systemText });
                    }
                     messages[firstUserMsgIndex].content = content;
                }
            } else {
                // If no user message, this system prompt might be ignored or cause issues.
                // Or create a dummy user message with the system prompt
                history.push({role: 'user', parts: [{text: systemText}]});
            }
         }
      }
    }

    const operationalMessages = messages.filter(msg => msg.role !== 'system');

    for (const msg of operationalMessages) {
      const geminiRole = msg.role === 'assistant' ? 'model' : 'user';
      const currentMessageParts = [];

      if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === 'text') {
            currentMessageParts.push({ text: part.text });
          } else if (part.type === 'image_url') {
            const { base64Data, mimeType } = await this._fetchAndBase64Image(part.image_url.url);
            currentMessageParts.push({ inlineData: { mimeType: mimeType, data: base64Data } });
          } else if (part.type === 'tool_output' && msg.role === 'tool' && msg.tool_call_id) {
            currentMessageParts.push({ functionResponse: { name: msg.name, response: part.content } });
          }
        }
      } else if (typeof msg.content === 'string') {
        currentMessageParts.push({ text: msg.content });
      }

      if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
        msg.tool_calls.forEach(tc => {
          try {
            currentMessageParts.push({ functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments || '{}') } });
          } catch (e) {
            throw new LLMPlugRequestError(`Failed to parse tool call arguments for ${tc.function.name}: ${e.message}. Arguments: ${tc.function.arguments}`, this.providerName, e);
          }
        });
      }

      if (currentMessageParts.length > 0) {
        // Ensure alternating roles if possible, though Gemini SDK is somewhat robust
        if (history.length > 0 && history[history.length - 1].role === geminiRole) {
            // If last role is same, merge parts (simplistic merge)
            // This is a basic fix; more complex merging might be needed for perfect conversation flow.
            history[history.length - 1].parts.push(...currentMessageParts);
        } else {
            history.push({ role: geminiRole, parts: currentMessageParts });
        }
      } else if (geminiRole === 'model' && msg.content === null && (!msg.tool_calls || msg.tool_calls.length === 0)) {
        history.push({ role: 'model', parts: [] });
      }
    }
    return { systemInstruction: this.defaultModel.startsWith('gemini-1.5') ? systemInstruction : undefined, history };
  }

  _prepareInputAsMessages(input) {
    if (typeof input === 'string') {
      return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input)) {
      return input;
    }
    throw new LLMPlugRequestError("Invalid input type. Must be a string or an array of ChatMessage objects.", this.providerName);
  }

  async generate(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    // For generate, which often implies a single prompt, we might use less strict safety for simple tests if configured
    const effectiveSafetySettings = options.extraParams?.usePermissiveSafety 
        ? this.permissiveSafetySettings 
        : (options.extraParams?.safetySettings || this.defaultSafetySettings);

    return this.chat(messages, { ...options, extraParams: { ...options.extraParams, safetySettings: effectiveSafetySettings } });
  }

  async chat(messages, options = {}) {
    const modelName = options.model || this.defaultModel;
    // Use specified safetySettings from options, or default.
    const safetySettingsToUse = options.extraParams?.safetySettings || this.defaultSafetySettings;
    const { systemInstruction, history } = await this._prepareMessages(messages);

    const toolsForGemini = options.tools?.map(tool => ({
        functionDeclarations: [{
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters ? { type: FunctionDeclarationSchemaType.OBJECT, ...tool.function.parameters } : undefined,
        }]
    }));

    const modelInstance = this.genAI.getGenerativeModel({
      model: modelName,
      safetySettings: safetySettingsToUse, // Apply chosen safety settings
      tools: toolsForGemini,
      systemInstruction: systemInstruction, // This is for gemini-1.5+
    });

    if (history.length === 0 && !systemInstruction && !this.defaultModel.startsWith('gemini-1.5')) { // 1.5 can take only systemInstruction
        throw new LLMPlugRequestError("Google AI chat requires at least one message.", this.providerName);
    }
    
    // If history is empty but systemInstruction exists for 1.5 models, it's valid
    if (history.length === 0 && this.defaultModel.startsWith('gemini-1.5') && !systemInstruction) {
         throw new LLMPlugRequestError("Google AI (1.5+) chat requires at least one message or a system instruction.", this.providerName);
    }


    try {
      const generationConfig = {
        temperature: options.temperature,
        maxOutputTokens: options.maxTokens,
        stopSequences: options.stopSequences,
        responseMimeType: options.responseFormat?.type === 'json_object' ? 'application/json' : undefined,
        ...(options.extraParams?.generationConfig || {}),
      };
      
      const result = await modelInstance.generateContent({
        contents: history,
        generationConfig: generationConfig,
      });

      const response = result.response;
      const candidate = response.candidates?.[0];

      let textContent = null;
      let toolCalls = [];

      if (candidate && candidate.content && candidate.content.parts) {
        textContent = candidate.content.parts
          .filter(part => part.text)
          .map(part => part.text)
          .join('')
          .trim();
        if (textContent === '') textContent = null; // Treat empty string as null for consistency

        toolCalls = candidate.content.parts
          .filter(part => part.functionCall)
          .map((part, index) => ({
            id: `gemini-fc-${Date.now()}-${index}`,
            type: 'function',
            function: {
              name: part.functionCall.name,
              arguments: JSON.stringify(part.functionCall.args || {}),
            },
          }));
      }
      
      const finishReason = candidate?.finishReason;
      const safetyRatings = candidate?.safetyRatings;

      // If no text and no tool calls, but a finish reason exists (like SAFETY), log it.
      if (textContent === null && toolCalls.length === 0 && finishReason) {
        console.warn(`[${this.providerName}] Null response. Finish Reason: ${finishReason}. Safety Ratings: ${JSON.stringify(safetyRatings)}`);
      }
      // If still null, and prompt was for code, it's highly likely safety.
      if (textContent === null && messages.some(m=>m.content.toString().toLowerCase().includes("function")) && finishReason === "SAFETY") {
          console.warn(`[${this.providerName}] Code generation might have been blocked by safety filters. Consider adjusting safety settings if appropriate.`);
      }


      const usage = {
        promptTokens: response.usageMetadata?.promptTokenCount,
        completionTokens: response.usageMetadata?.candidatesTokenCount || candidate?.tokenCount,
        totalTokens: response.usageMetadata?.totalTokenCount,
      };
      
      return {
        text: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: usage,
        finishReason: finishReason,
        rawResponse: response, // Include full response for debugging
      };
    } catch (error) {
      // Check if the error is a GoogleGenerativeAIResponseError and log details
      if (error.message.includes("GoogleGenerativeAI Error") && error.message.includes("response data")) {
           console.error(`[${this.providerName}] API Error Details:`, error.message);
      }
      throw new LLMPlugRequestError(`Google AI API chat request failed for model ${modelName}: ${error.message}`, this.providerName, error);
    }
  }

  async *generateStream(input, options = {}) {
    const messages = this._prepareInputAsMessages(input);
    // For generate, which often implies a single prompt, we might use less strict safety for simple tests if configured
    const effectiveSafetySettings = options.extraParams?.usePermissiveSafety 
        ? this.permissiveSafetySettings 
        : (options.extraParams?.safetySettings || this.defaultSafetySettings);
    yield* this.chatStream(messages, { ...options, extraParams: { ...options.extraParams, safetySettings: effectiveSafetySettings } });
  }

  async *chatStream(messages, options = {}) {
    const modelName = options.model || this.defaultModel;
    const safetySettingsToUse = options.extraParams?.safetySettings || this.defaultSafetySettings;
    const { systemInstruction, history } = await this._prepareMessages(messages);

    const toolsForGemini = options.tools?.map(tool => ({
        functionDeclarations: [{
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters ? { type: FunctionDeclarationSchemaType.OBJECT, ...tool.function.parameters } : undefined,
        }]
    }));

    const modelInstance = this.genAI.getGenerativeModel({
      model: modelName,
      safetySettings: safetySettingsToUse,
      tools: toolsForGemini,
      systemInstruction: systemInstruction,
    });

    if (history.length === 0 && !systemInstruction && !this.defaultModel.startsWith('gemini-1.5')) {
        throw new LLMPlugRequestError("Google AI chat stream requires at least one message.", this.providerName);
    }
    if (history.length === 0 && this.defaultModel.startsWith('gemini-1.5') && !systemInstruction) {
         throw new LLMPlugRequestError("Google AI (1.5+) chat stream requires at least one message or a system instruction.", this.providerName);
    }


    const generationConfig = {
        temperature: options.temperature,
        maxOutputTokens: options.maxTokens,
        stopSequences: options.stopSequences,
        responseMimeType: options.responseFormat?.type === 'json_object' ? 'application/json' : undefined,
        ...(options.extraParams?.generationConfig || {}),
    };

    try {
      const result = await modelInstance.generateContentStream({
        contents: history,
        generationConfig: generationConfig,
      });

      for await (const chunk of result.stream) {
        const chunkData = { rawChunk: chunk };
        const candidate = chunk.candidates?.[0];

        if (candidate?.content?.parts) {
          const textPart = candidate.content.parts.find(part => part.text);
          if (textPart?.text) {
            chunkData.text = textPart.text;
          }

          const functionCallParts = candidate.content.parts.filter(part => part.functionCall);
          if (functionCallParts.length > 0) {
            chunkData.toolCalls = functionCallParts.map((part, index) => ({
              id: `gemini-fc-stream-${Date.now()}-${index}`,
              type: 'function',
              function: { name: part.functionCall.name, arguments: JSON.stringify(part.functionCall.args || {}) }
            }));
          }
        }
        
        if (chunk.usageMetadata) {
            chunkData.usage = {
                promptTokens: chunk.usageMetadata.promptTokenCount,
                completionTokens: chunk.usageMetadata.candidatesTokenCount || candidate?.tokenCount,
                totalTokens: chunk.usageMetadata.totalTokenCount,
            };
        }

        if (candidate?.finishReason) {
            chunkData.finishReason = candidate.finishReason;
            if (chunkData.finishReason === "SAFETY" || (chunkData.finishReason && !chunkData.text && (!chunkData.toolCalls || chunkData.toolCalls.length === 0))) {
                console.warn(`[${this.providerName} Stream] Chunk with Finish Reason: ${chunkData.finishReason}. Safety Ratings: ${JSON.stringify(candidate.safetyRatings)}`);
            }
        }
        yield chunkData;
      }
    } catch (error) {
      throw new LLMPlugRequestError(`Google AI API chat stream failed for model ${modelName}: ${error.message}`, this.providerName, error);
    }
  }
}