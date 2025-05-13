import { LLMPlug } from '../src/index.js'; // If running from llmplug root
// Or, if llmplug is installed as a dependency:
// import { LLMPlug } from 'llmplug';

// Ensure you have a .env file in the root of your project with API keys like:
// OPENAI_API_KEY=your_openai_key
// ANTHROPIC_API_KEY=your_anthropic_key
// GOOGLE_GEMINI_API_KEY=your_google_key
// HUGGINGFACE_API_TOKEN=your_hf_token (for certain models)

// To run this example from the llmplug root:
// node --env-file=.env examples/simpleUsage.js

async function runSimpleExamples() {
  console.log("--- LLMPlug Simple Usage ---");

  // 1. OpenAI Example (Simple Text Generation)
  try {
    const openai = LLMPlug.getProvider('openai'); // Uses default model (gpt-3.5-turbo)
    const prompt = "What are three fun facts about the ocean?";
    console.log("\n[OpenAI] Generating response for:", prompt);

    const result = await openai.generate(prompt, { maxTokens: 500 });
    // The 'result' object contains: result.text, result.usage, result.finishReason, result.rawResponse
    console.log("[OpenAI] Response:", result.text);
    console.log("[OpenAI] Finish Reason:", result.finishReason);

  } catch (error) {
    console.error("[OpenAI] Error:", error.message);
  }

  // 2. Anthropic Example (Simple Chat)
  try {
    const anthropic = LLMPlug.getProvider('anthropic'); // Uses default model (e.g., claude-3-haiku)
    const messages = [
      { role: 'user', content: "Write a short haiku about a sleeping cat." }
    ];
    console.log("\n[Anthropic] Sending chat messages:", messages);

    const result = await anthropic.chat(messages, { maxTokens: 500 });
    console.log("[Anthropic] Response:", result.text);
    console.log("[Anthropic] Usage (Output Tokens):", result.usage?.completionTokens);

  } catch (error) {
    console.error("[Anthropic] Error:", error.message);
  }

  // 3. Google Gemini Example (Simple Text Generation with a specific model)
  try {
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-2.0-pro-exp-02-05' }); // Specifying a model
    const codePrompt = "Write a simple Python function to add two numbers.";
    console.log("\n[Google Gemini] Generating response for:", codePrompt);

    const result = await google.generate(codePrompt, { maxTokens: 2500, temperature: 0.3 });
    console.log("[Google Gemini] Response:\n", result.text);

  } catch (error) {
    console.error("[Google Gemini] Error:", error.message);
  }
  
  // 4. Hugging Face Example (Requires modelId in config)
  try {
    // For Hugging Face, modelId is REQUIRED.
    // Using a smaller, generally available model for quick test.
    const hf = LLMPlug.getProvider('huggingface', { modelId: 'gpt4.5' }); 
    const hfPrompt = "Once upon a time, in a land of code,";
    console.log("\n[Hugging Face] Generating response for:", hfPrompt);

    const result = await hf.generate(hfPrompt, { maxTokens: 500 });
    console.log("[Hugging Face] Response:", result.text);
    // Note: Hugging Face Inference API often doesn't provide usage/finishReason.

  } catch (error)
  {
    console.error("[Hugging Face] Error:", error.message);
  }

  console.log("\n--- Simple Usage Examples Complete ---");
}

runSimpleExamples().catch(error => {
  console.error("Unhandled error in simple examples:", error.message);
});