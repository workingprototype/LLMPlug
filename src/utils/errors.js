
export class LLMPlugError extends Error {
  constructor(message, provider, originalError = null) {
    super(message);
    this.name = 'LLMPlugError';
    this.provider = provider;
    this.originalError = originalError;
    if (originalError && originalError.stack) {
      this.stack = `${this.stack}\nCaused by: ${originalError.stack}`;
    }
  }
}

export class LLMPlugConfigurationError extends LLMPlugError {
  constructor(message, provider) {
    super(message, provider);
    this.name = 'LLMPlugConfigurationError';
  }
}

export class LLMPlugRequestError extends LLMPlugError {
  constructor(message, provider, originalError = null) {
    super(message, provider, originalError);
    this.name = 'LLMPlugRequestError';
  }
}

export class LLMPlugToolError extends LLMPlugError {
  constructor(message, provider, toolName, originalError = null) {
    super(message, provider, originalError);
    this.name = 'LLMPlugToolError';
    this.toolName = toolName;
  }
}