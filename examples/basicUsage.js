import { LLMPlug, LLMPlugError } from '../src/index.js'; // if running from root of llmplug
// or from an installed package: import { LLMPlug, LLMPlugError } from 'llmplug';

async function main() {
  console.log("LLMPlug Basic Usage Example\n");

  // --- OpenAI Example ---
  try {
    console.log("--- OpenAI ---");
    // API key can be in .env (OPENAI_API_KEY) or passed in config
    const openai = LLMPlug.getProvider('openai', { defaultModel: 'gpt-3.5-turbo' }); 
    
    const Prompt = "Give 3 tips for a healthy lifestyle.";
    console.log(`OpenAI Generate Prompt (Kannada): "${Prompt}"`);
    let response = await openai.generate(Prompt, { maxTokens: 150 });
    console.log("OpenAI Generate Response:", response);

    const chatMessages = [
      { role: 'system', content: 'You are a helpful assistant that speaks like a pirate.' },
      { role: 'user', content: 'Hello, who are you?' }
    ];
    console.log("\nOpenAI Chat Messages:", chatMessages);
    response = await openai.chat(chatMessages, { maxTokens: 100 });
    console.log("OpenAI Chat Response:", response);

  } catch (error) {
    console.error("OpenAI Error:", error.message);
    if (error.originalError) console.error("Original Error:", error.originalError.message);
  }

  // --- Anthropic Example ---
  try {
    console.log("\n--- Anthropic ---");
    // API key can be in .env (ANTHROPIC_API_KEY) or passed in config
    const anthropic = LLMPlug.getProvider('anthropic', { defaultModel: 'claude-3-haiku-20240307' }); 

    const storyPrompt = "Write a very short story about a curious robot discovering a flower.";
    console.log(`Anthropic Generate Prompt: "${storyPrompt}"`);
    // Anthropic generate uses chat underneath for claude-3, so options are similar
    let response = await anthropic.generate(storyPrompt, { maxTokens: 200 });
    console.log("Anthropic Generate Response:", response);

    const anthropicChat = [
      { role: 'user', content: "What's the largest animal on Earth?" }
    ];
    console.log("\nAnthropic Chat Messages:", anthropicChat);
    response = await anthropic.chat(anthropicChat, { maxTokens: 100 });
    console.log("Anthropic Chat Response:", response);
    
  } catch (error) {
    console.error("Anthropic Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) {
        console.error("Original Anthropic Error Details:", error.originalError);
    }
  }

  // --- Google Gemini Example ---
  try {
    console.log("\n--- Google Gemini ---");
    // API key can be in .env (GOOGLE_GEMINI_API_KEY) or passed in config
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-2.0-pro-exp-02-05' }); 

    const codePrompt = "Write a python function to reverse a string.";
    console.log(`Google Generate Prompt: "${codePrompt}"`);
    let response = await google.generate(codePrompt, { maxTokens: 150 });
    console.log("Google Generate Response:\n", response);

    const googleChat = [
      { role: 'user', content: "What are three fun facts about the planet Mars?" }
    ];
    console.log("\nGoogle Chat Messages:", googleChat);
    response = await google.chat(googleChat, { maxTokens: 200 });
    console.log("Google Chat Response:\n", response);

  } catch (error) {
    console.error("Google Gemini Error:", error.message);
    if (error.originalError) console.error("Original Error:", error.originalError.message);
  }

  // --- Hugging Face Example ---
  try {
    console.log("\n--- Hugging Face ---");
    // API token can be in .env (HUGGINGFACE_API_TOKEN) or passed in config
    // modelId IS REQUIRED for Hugging Face
    const hf = LLMPlug.getProvider('huggingface', { 
      modelId: 'mistralai/Mistral-7B-Instruct-v0.1', // A popular open model
      // modelId: 'gpt2', // Simpler model, faster, but less capable
      // task: 'text-generation' // default
    }); 

    const hfPrompt = "Once upon a time, in a land full of code,";
    console.log(`Hugging Face Generate Prompt: "${hfPrompt}"`);
    let response = await hf.generate(hfPrompt, { maxTokens: 50 });
    console.log("Hugging Face Generate Response:", response);

    // Conversational for HF is a bit more specific
    const hfChat = [
      { role: 'system', content: "You are a witty and slightly sarcastic chatbot." },
      { role: 'user', content: 'What is the meaning of life?' },
      // { role: 'assistant', content: "42, obviously. But what's the question again?"}, // example of history
      // { role: 'user', content: "That's not very helpful."}
    ];
    console.log("\nHugging Face Chat Messages:", hfChat);
    response = await hf.chat(hfChat, { maxTokens: 60 });
    console.log("Hugging Face Chat Response:", response);
    
  } catch (error) {
    console.error("Hugging Face Error:", error.message);
    if (error.originalError) console.error("Original Error:", error.originalError.message);
  }
}

main().catch(console.error);