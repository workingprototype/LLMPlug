import { LLMPlug } from '../src/index.js'; // If running from llmplug root
// Or, if llmplug is installed as a dependency:
// import { LLMPlug } from 'llmplug';

// To run this example from the llmplug root:
// node --env-file=.env examples/simpleUsage.js
// (Ensure your .env file has API keys for cloud providers if testing them)

// For local providers (Ollama, LlamaCpp, Oobabooga), ensure the respective
// server is running locally and the model is available.

async function runSimpleExamples() {
  console.log("--- LLMPlug Simple Usage ---");

  // --- Cloud Providers (require API keys in .env) ---
  try {
    const openai = LLMPlug.getProvider('openai');
    const prompt = "What are three fun facts about the Moon?";
    console.log("\n[OpenAI] Generating response for:", prompt);
    const result = await openai.generate(prompt, { maxTokens: 100 });
    console.log("[OpenAI] Response:", result.text);
  } catch (error) {
    console.warn("[OpenAI] Error:", error.message, "(Skipping - ensure API key is set)");
  }

  try {
    const anthropic = LLMPlug.getProvider('anthropic');
    const messages = [{ role: 'user', content: "Write a short haiku about a rainy day." }];
    console.log("\n[Anthropic] Sending chat messages for haiku...");
    const result = await anthropic.chat(messages, { maxTokens: 60 });
    console.log("[Anthropic] Response:", result.text);
  } catch (error) {
    console.warn("[Anthropic] Error:", error.message, "(Skipping - ensure API key is set)");
  }

  try {
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-1.0-pro' });
    const codePrompt = "Write a simple JavaScript function to greet a user by name.";
    console.log("\n[Google Gemini] Generating response for:", codePrompt);
    const result = await google.generate(codePrompt, { maxTokens: 120, temperature: 0.3 });
    console.log("[Google Gemini] Response:\n", result.text);
  } catch (error) {
    console.warn("[Google Gemini] Error:", error.message, "(Skipping - ensure API key is set or try permissive safety for code)");
  }
  
  try {
    const cohere = LLMPlug.getProvider('cohere');
    const coherePrompt = "Summarize the concept of photosynthesis in one sentence.";
    console.log("\n[Cohere] Generating response for:", coherePrompt);
    const result = await cohere.generate(coherePrompt, { maxTokens: 80 });
    console.log("[Cohere] Response:", result.text);
  } catch (error) {
    console.warn("[Cohere] Error:", error.message, "(Skipping - ensure API key is set)");
  }

  try {
    const mistralai = LLMPlug.getProvider('mistralai', {defaultModel: 'open-mistral-7b'}); // Use a smaller, faster model
    const mistralPrompt = "Explain what an API is in simple terms.";
    console.log("\n[Mistral AI] Generating response for:", mistralPrompt);
    const result = await mistralai.generate(mistralPrompt, { maxTokens: 100 });
    console.log("[Mistral AI] Response:", result.text);
  } catch (error) {
    console.warn("[Mistral AI] Error:", error.message, "(Skipping - ensure API key is set)");
  }
  
  try {
    // OpenRouter requires a model to be specified, e.g., a free or common one.
    // You'll need an OPENROUTER_API_KEY.
    // Find models at https://openrouter.ai/models
    const openrouter = LLMPlug.getProvider('openrouter', { 
        defaultModel: 'nousresearch/nous-capybara-7b-v1.9', // Example free model
        // httpReferer: 'YOUR_SITE_URL', // Optional, but recommended by OpenRouter
        // xTitle: 'LLMPlug Simple Test',    // Optional
    });
    const orPrompt = "What is the capital of Australia?";
    console.log("\n[OpenRouter] Generating with model 'nousresearch/nous-capybara-7b-v1.9':", orPrompt);
    const result = await openrouter.generate(orPrompt, { maxTokens: 50 });
    console.log("[OpenRouter] Response:", result.text);
  } catch (error) {
    console.warn("[OpenRouter] Error:", error.message, "(Skipping - ensure API key and model are set, and model is valid)");
  }


  // --- Local Providers (require local server running) ---
  console.log("\n--- Local Provider Examples (Ensure Servers are Running) ---");

  // 1. Ollama Example
  //    - Make sure Ollama is running (e.g., `ollama serve`)
  //    - Make sure you have pulled a model (e.g., `ollama pull llama3:8b`)
  const ollamaModel = "llama3:8b"; // Change to a model you have pulled
  try {
    const ollama = LLMPlug.getProvider('ollama', {
      // baseURL: "http://localhost:11434/v1", // Default, can override if needed
      defaultModel: ollamaModel 
    });
    const ollamaPrompt = `Tell me a joke about computers. (Using Ollama with ${ollamaModel})`;
    console.log(`\n[Ollama] Generating with model '${ollamaModel}':`, ollamaPrompt);
    
    // You can also list models:
    // const localModels = await ollama.listLocalModels();
    // console.log(`[Ollama] Available local models: ${localModels.join(', ')}`);

    const result = await ollama.generate(ollamaPrompt, { maxTokens: 80 });
    console.log("[Ollama] Response:", result.text);
  } catch (error) {
    console.warn(`[Ollama] Error with model '${ollamaModel}':`, error.message, "(Skipping - ensure Ollama server is running and model is pulled)");
  }

  // 2. Llama.cpp Server Example
  //    - Make sure your llama.cpp server is running with an OpenAI-compatible API
  //      (e.g., `./server -m your_model.gguf -c 2048 --port 8080 --host 0.0.0.0`)
  //    - The 'defaultModel' here can often be a dummy string if the server serves one model,
  //      or it should match an alias if the server handles multiple.
  const llamaCppModel = "local-llama-cpp-model"; // Placeholder, actual model name depends on server setup
  try {
    const llamaCpp = LLMPlug.getProvider('llamacpp', {
      baseURL: "http://localhost:8080/v1", // Adjust if your server runs on a different port
      defaultModel: llamaCppModel 
    });
    const llamaCppPrompt = `What is 2 + 2? (Using Llama.cpp server with model ${llamaCppModel})`;
    console.log(`\n[Llama.cpp Server] Generating with model '${llamaCppModel}':`, llamaCppPrompt);
    const result = await llamaCpp.generate(llamaCppPrompt, { maxTokens: 30 });
    console.log("[Llama.cpp Server] Response:", result.text);
  } catch (error) {
    console.warn(`[Llama.cpp Server] Error with model '${llamaCppModel}':`, error.message, "(Skipping - ensure llama.cpp server is running with OpenAI API and model is loaded)");
  }

  // 3. Oobabooga (Text Generation WebUI) Example
  //    - Make sure Oobabooga is running with the OpenAI API extension enabled.
  //    - The model loaded in Oobabooga will be used.
  const oobaboogaModel = "oobabooga-current-model"; // Placeholder, usually uses the loaded model
  try {
    const oobabooga = LLMPlug.getProvider('oobabooga', {
      baseURL: "http://localhost:5000/v1", // Default for Oobabooga OpenAI extension
      defaultModel: oobaboogaModel 
    });
    const oobaboogaPrompt = `Write a single sentence about a curious robot. (Using Oobabooga with ${oobaboogaModel})`;
    console.log(`\n[Oobabooga] Generating with model '${oobaboogaModel}':`, oobaboogaPrompt);
    const result = await oobabooga.generate(oobaboogaPrompt, { maxTokens: 40 });
    console.log("[Oobabooga] Response:", result.text);
  } catch (error) {
    console.warn(`[Oobabooga] Error with model '${oobaboogaModel}':`, error.message, "(Skipping - ensure Oobabooga is running with OpenAI extension enabled)");
  }
  
  // 4. Hugging Face (already present, keeping for completeness)
  try {
    const hf = LLMPlug.getProvider('huggingface', { modelId: 'gpt2' }); 
    const hfPrompt = "A short poem about the dawn:";
    console.log("\n[Hugging Face] Generating response for:", hfPrompt);
    const result = await hf.generate(hfPrompt, { maxTokens: 40 });
    console.log("[Hugging Face] Response:", result.text);
  } catch (error) {
    console.warn("[Hugging Face] Error:", error.message, "(Skipping - ensure API token if needed for model)");
  }


  console.log("\n--- Simple Usage Examples Complete ---");
}

runSimpleExamples().catch(error => {
  console.error("Unhandled error in simple examples:", error.message);
});