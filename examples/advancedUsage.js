import { LLMPlug, LLMPlugError, LLMPlugToolError } from '../src/index.js';
import readline from 'node:readline/promises';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function main() {
  console.log("LLMPlug Advanced Usage Showcase\n");

  const displayResult = (providerName, type, result, note = "") => {
    console.log(`\n--- ${providerName} | ${type} ${note} ---`);
    if (result.text !== null && result.text !== undefined) console.log("Text:", result.text.trim());
    if (result.toolCalls && result.toolCalls.length > 0) {
      console.log("Tool Calls Requested:");
      result.toolCalls.forEach(tc => console.log(`  - ID: ${tc.id}, Function: ${tc.function.name}(${tc.function.arguments})`));
    }
    if (result.usage) console.log("Usage:", `Prompt: ${result.usage.promptTokens || 'N/A'}, Completion: ${result.usage.completionTokens || 'N/A'}, Total: ${result.usage.totalTokens || 'N/A'}`);
    if (result.finishReason) console.log("Finish Reason:", result.finishReason);
    if (result.rawResponse?.candidates?.[0]?.safetyRatings) console.log("Safety Ratings (Gemini):", JSON.stringify(result.rawResponse.candidates[0].safetyRatings));
    console.log("-----------------------------------\n");
  };

  const handleStream = async (providerName, type, stream, note = "") => {
    console.log(`\n--- ${providerName} | ${type} (Streaming) ${note} ---`);
    let fullText = '';
    const toolCalls = {};
    let finalUsage, finalFinishReason, lastSafetyRatings;
    process.stdout.write("Streamed Text: ");
    for await (const chunk of stream) {
      if (chunk.text) { process.stdout.write(chunk.text); fullText += chunk.text; }
      if (chunk.toolCalls) {
        chunk.toolCalls.forEach(tcChunk => {
          if (!toolCalls[tcChunk.id]) toolCalls[tcChunk.id] = { ...tcChunk, function: { name: tcChunk.function.name, arguments: '' } };
          if (tcChunk.function.arguments) toolCalls[tcChunk.id].function.arguments += tcChunk.function.arguments;
        });
      }
      if (chunk.usage) finalUsage = chunk.usage;
      if (chunk.finishReason) finalFinishReason = chunk.finishReason;
      if (chunk.rawChunk?.candidates?.[0]?.safetyRatings) lastSafetyRatings = chunk.rawChunk.candidates[0].safetyRatings;
    }
    process.stdout.write("\n");
    console.log("Full Streamed Text:", fullText.trim());
    if (Object.keys(toolCalls).length > 0) {
      console.log("Streamed Tool Calls Requested:");
      Object.values(toolCalls).forEach(tc => console.log(`  - ID: ${tc.id}, Function: ${tc.function.name}(${tc.function.arguments})`));
    }
    if (finalUsage) console.log("Stream Usage:", `Prompt: ${finalUsage.promptTokens || 'N/A'}, Completion: ${finalUsage.completionTokens || 'N/A'}, Total: ${finalUsage.totalTokens || 'N/A'}`);
    if (finalFinishReason) console.log("Stream Finish Reason:", finalFinishReason);
    if (lastSafetyRatings) console.log("Stream Safety Ratings (Gemini - last chunk):", JSON.stringify(lastSafetyRatings));
    console.log("-----------------------------------\n");
    return { text: fullText.trim(), toolCalls: Object.values(toolCalls), usage: finalUsage, finishReason: finalFinishReason };
  };

  const tools = [
    { type: 'function', function: { name: 'get_current_weather', description: 'Get current weather. Use for weather queries.', parameters: { type: 'object', properties: { location: { type: 'string', description: 'City and state, e.g. San Francisco, CA' }, unit: { type: 'string', enum: ['celsius', 'fahrenheit'] } }, required: ['location'] } } },
    { type: 'function', function: { name: 'lookup_stock_price', description: 'Get current stock price. Use for stock queries.', parameters: { type: 'object', properties: { ticker_symbol: { type: 'string', description: 'Stock ticker, e.g., AAPL' } }, required: ['ticker_symbol'] } } },
    { type: 'function', function: { name: 'generate_python_code', description: 'Generates Python code. Returns code as string.', parameters: { type: 'object', properties: { task_description: { type: 'string', description: 'Python task description.' } }, required: ['task_description']}}}
  ];
  const availableTools = {
    get_current_weather: async ({ location, unit = 'fahrenheit' }) => { console.log(` MOCK TOOL: 'get_current_weather' for ${location}, ${unit}`); await new Promise(r => setTimeout(r,100)); return { temperature: '20C/68F', conditions: 'partly cloudy' }; },
    lookup_stock_price: async ({ ticker_symbol }) => { console.log(` MOCK TOOL: 'lookup_stock_price' for ${ticker_symbol}`); await new Promise(r => setTimeout(r,100)); return { ticker: ticker_symbol.toUpperCase(), price: (Math.random() * 400 + 20).toFixed(2), currency: "USD" }; },
    generate_python_code: async ({ task_description }) => { console.log(` MOCK TOOL: 'generate_python_code' for: "${task_description}"`); await new Promise(r => setTimeout(r,50)); return `print("Mock code for: ${task_description.replace(/"/g, '\\"')}")\n# TODO: Implement logic`; }
  };
  async function processToolCalls(provider, conversationHistory, toolCalls) { /* ... (same as previous advancedUsage.js) ... */ 
    if (!toolCalls || toolCalls.length === 0) return conversationHistory;
    let updatedConversation = [...conversationHistory];
    updatedConversation.push({ role: 'assistant', content: null, tool_calls: toolCalls });
    for (const toolCall of toolCalls) {
      const functionName = toolCall.function.name;
      let functionArgs;
      try { functionArgs = JSON.parse(toolCall.function.arguments || '{}'); } 
      catch (e) {
        updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: "Invalid JSON arguments", details: e.message } }] });
        continue;
      }
      if (availableTools[functionName]) {
        try {
          const toolOutputContent = await availableTools[functionName](functionArgs);
          updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: toolOutputContent }] });
        } catch (toolError) {
          updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: toolError.message } }] });
        }
      } else {
         updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: `Tool ${functionName} not available.` } }] });
      }
    }
    return updatedConversation;
  }

  const catImageUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg";
  const logoImageUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1200px-React-icon.svg.png";

  // --- Cloud Providers ---
  try { /* OpenAI section ... (can keep as is from previous advancedUsage.js) */
    console.log("===== OpenAI =====");
    const openai = LLMPlug.getProvider('openai', { defaultModel: 'gpt-4o' });
    let openAIResult = await openai.chat(
        [{ role: 'user', content: [{ type: 'text', text: 'Describe this cat image concisely:' }, { type: 'image_url', image_url: { url: catImageUrl, detail: 'low' } }]}],
        { maxTokens: 150 }
    );
    displayResult("OpenAI", "Multimodal Chat", openAIResult);
    let toolConversation = [{role: 'system', content: 'Use tools effectively.'}, { role: 'user', content: "Weather in London and GOOGL stock?" }];
    for (let i = 0; i < 3; i++) {
        const toolCallResult = await openai.chat(toolConversation, { tools: tools, maxTokens: 300 });
        displayResult("OpenAI", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls?.length) toolConversation = await processToolCalls(openai, toolConversation, toolCallResult.toolCalls); else break;
        if (toolCallResult.text && i > 0) break;
    }
  } catch (e) { console.warn("[OpenAI] Skipping:", e.message); }
  
  try { /* Anthropic section ... (can keep as is) */
    console.log("\n===== Anthropic =====");
    const anthropic = LLMPlug.getProvider('anthropic', { defaultModel: 'claude-3-sonnet-20240229' });
    let anthropicResult = await anthropic.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'Explain this logo:' }, { type: 'image_url', image_url: { url: logoImageUrl } }] }], { maxTokens: 200 }
    );
    displayResult("Anthropic", "Multimodal Chat", anthropicResult);
    let toolConversation = [{role: 'user', content: "Weather in Berlin and Python code for string reversal?" }];
    for (let i = 0; i < 3; i++) {
        const toolCallResult = await anthropic.chat(toolConversation, { tools: tools, maxTokens: 350 });
        displayResult("Anthropic", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls?.length) toolConversation = await processToolCalls(anthropic, toolConversation, toolCallResult.toolCalls); else break;
        if (toolCallResult.text && i > 0) break;
    }
  } catch (e) { console.warn("[Anthropic] Skipping:", e.message); }

  try { /* Google Gemini section ... (can keep as is, including safety retry) */
    console.log("\n===== Google Gemini =====");
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-1.5-pro-latest' });
    let geminiResult = await google.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'Funny caption for this cat:' }, { type: 'image_url', image_url: { url: catImageUrl } }] }], { maxTokens: 200 }
    );
    displayResult("Google Gemini", "Multimodal Chat", geminiResult);
    let toolConversation = [{role: 'system', content: 'You MUST use tools if available.'}, { role: 'user', content: "Weather in Rome? Python code for prime check?" }];
    let geminiSafetyRetry = false;
     for (let i = 0; i < 3; i++) {
        const toolCallOptions = { tools: tools, maxTokens: 300, extraParams: { usePermissiveSafety: geminiSafetyRetry } };
        const toolCallResult = await google.chat(toolConversation, toolCallOptions);
        displayResult("Google Gemini", `Tool Call Step ${i + 1}`, toolCallResult, geminiSafetyRetry ? '(Permissive Safety)' : '');
        if (toolCallResult.text === null && toolCallResult.finishReason === 'SAFETY' && !geminiSafetyRetry) {
            console.warn("Gemini: Retrying with permissive safety ONCE."); geminiSafetyRetry = true; i--; continue;
        }
        geminiSafetyRetry = false;
        if (toolCallResult.toolCalls?.length) toolConversation = await processToolCalls(google, toolConversation, toolCallResult.toolCalls); else break;
        if (toolCallResult.text && i > 0) break;
    }
  } catch (e) { console.warn("[Google Gemini] Skipping:", e.message); }

  // --- New Cloud Providers ---
  try {
    console.log("\n===== Cohere =====");
    const cohere = LLMPlug.getProvider('cohere', { defaultChatModel: 'command-r-plus' }); // command-r-plus is good for tool use
    let cohereResult = await cohere.generate("Write a tagline for a new AI-powered coffee machine.", { maxTokens: 50 });
    displayResult("Cohere", "Generate", cohereResult);
    
    let toolConversation = [
        {role: 'system', content: 'You are Command-R-Plus, a helpful AI. Use tools to answer questions.'},
        {role: 'user', content: "What's the weather in Toronto and the stock price for MSFT?"}
    ];
    for (let i = 0; i < 3; i++) {
        const toolCallResult = await cohere.chat(toolConversation, { tools: tools, maxTokens: 300 });
        displayResult("Cohere", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls?.length) {
            // Cohere's tool result formatting is specific. Our processToolCalls is generic.
            // For Cohere, the `processToolCalls` would need to ensure the 'tool' message sent back
            // has the `tool_results` structure Cohere expects. The provider attempts this.
            toolConversation = await processToolCalls(cohere, toolConversation, toolCallResult.toolCalls);
        } else break;
        if (toolCallResult.text && i > 0) break;
    }
    await handleStream("Cohere", "Chat Stream", cohere.chatStream([{role: 'user', content: "Explain the difference between RAM and ROM."}], {maxTokens: 200}));
  } catch (e) { console.warn("[Cohere] Skipping:", e.message); }

  try {
    console.log("\n===== Mistral AI =====");
    const mistral = LLMPlug.getProvider('mistralai', { defaultModel: 'mistral-large-latest' });
    let mistralResult = await mistral.chat(
        [{role: 'user', content: "What are the main features of the Mistral Large model?"}], { maxTokens: 250 }
    );
    displayResult("Mistral AI", "Chat", mistralResult);

    let toolConversation = [
        {role: 'system', content: 'You are a helpful assistant. Use tools when appropriate.'},
        {role: 'user', content: "Generate Python code for a fibonacci sequence function, then tell me the weather in Paris."}
    ];
    for (let i = 0; i < 3; i++) {
        const toolCallResult = await mistral.chat(toolConversation, { tools: tools, maxTokens: 400 });
        displayResult("Mistral AI", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls?.length) toolConversation = await processToolCalls(mistral, toolConversation, toolCallResult.toolCalls); else break;
        if (toolCallResult.text && i > 0) break;
    }
    await handleStream("Mistral AI", "Chat Stream (JSON mode attempt)", mistral.chatStream(
        [{role: 'system', content: 'Respond only in valid JSON.'}, {role: 'user', content: 'Give me a JSON object with city: Paris, country: France.'}],
        { responseFormat: {type: 'json_object'}, maxTokens: 100}
    ));
  } catch (e) { console.warn("[Mistral AI] Skipping:", e.message); }

  // --- OpenRouter (Aggregator) ---
  try {
    console.log("\n===== OpenRouter =====");
    // Find models: https://openrouter.ai/models - use "vendor/model-name" format
    // Example: 'mistralai/mistral-7b-instruct', 'google/gemini-pro', 'anthropic/claude-3-haiku'
    const openrouter = LLMPlug.getProvider('openrouter', {
        defaultModel: 'mistralai/mistral-7b-instruct-v0.2', // A generally good and often free/low-cost model
        // httpReferer: 'YOUR_SITE_URL', // Recommended by OpenRouter
        // xTitle: 'LLMPlug Advanced Test', // Recommended
    });
    let orResult = await openrouter.chat(
        [{role: 'user', content: `Tell me about the model ${openrouter.defaultModel} using OpenRouter.`}], { maxTokens: 200 }
    );
    displayResult("OpenRouter", "Chat", orResult);

    // Tool use with an OpenRouter model that supports it (e.g., a capable Mistral or OpenAI model)
    let orToolConversation = [{role: 'user', content: "What's the weather in New York City using tools?"}];
    const orToolModel = 'openai/gpt-3.5-turbo'; // Or another tool-capable model available on OR
    for (let i = 0; i < 2; i++) {
        const toolCallResult = await openrouter.chat(orToolConversation, { model: orToolModel, tools: tools, maxTokens: 250 });
        displayResult("OpenRouter", `Tool Call (${orToolModel}) Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls?.length) orToolConversation = await processToolCalls(openrouter, orToolConversation, toolCallResult.toolCalls); else break;
        if (toolCallResult.text && i > 0) break;
    }
  } catch (e) { console.warn("[OpenRouter] Skipping:", e.message); }


  // --- Local Providers ---
  console.log("\n--- Local Provider Advanced Examples (Ensure Servers & Models are Ready!) ---");
  const ollamaModelToTest = "llama3:8b"; // Ensure this model is pulled: `ollama pull llama3:8b`
  const llamaCppModelName = "local-llama-cpp"; // This is a placeholder for your loaded GGUF model alias
  const oobaModelName = "local-ooba-model"; // Placeholder

  try {
    console.log(`\n===== Ollama (Model: ${ollamaModelToTest}) =====`);
    const ollama = LLMPlug.getProvider('ollama', { defaultModel: ollamaModelToTest });
    // Test listing models
    try {
        const localModels = await ollama.listLocalModels();
        console.log(`[Ollama] Locally available models: ${localModels.join(', ') || 'None found (or error)'}`);
    } catch (listError) { console.warn("[Ollama] Could not list local models:", listError.message); }
    
    let ollamaResult = await ollama.chat(
        [{role: 'user', content: `Write a short Python script to list files in a directory using Ollama with ${ollamaModelToTest}. Be concise.`}],
        { maxTokens: 250, temperature: 0.5, extraParams: { ollamaOptions: { num_ctx: 4096 } } } // Example of passing native ollama option
    );
    displayResult("Ollama", "Chat (Code Gen)", ollamaResult);

    // Tool use with Ollama depends HEAVILY on the model.
    // Models like Llama3-Instruct are more likely to attempt it.
    let ollamaToolConversation = [{role: 'user', content: `Using model ${ollamaModelToTest}: what's the weather in Berlin?`}];
    if (ollamaModelToTest.includes("instruct") || ollamaModelToTest.includes("llama3")) { // Heuristic
        for (let i = 0; i < 2; i++) {
            const toolCallResult = await ollama.chat(ollamaToolConversation, { tools: tools, maxTokens: 300 });
            displayResult("Ollama", `Tool Call Step ${i + 1}`, toolCallResult);
            if (toolCallResult.toolCalls?.length) ollamaToolConversation = await processToolCalls(ollama, ollamaToolConversation, toolCallResult.toolCalls); else break;
            if (toolCallResult.text && i > 0) break;
        }
    } else { console.log(`[Ollama] Skipping tool use test for ${ollamaModelToTest} as it might not support it well.`); }
    await handleStream("Ollama", "Chat Stream", ollama.chatStream([{role: 'user', content: `Explain "localhost" using ${ollamaModelToTest}.`}], {maxTokens: 150}));

  } catch (e) { console.warn(`[Ollama] Skipping model ${ollamaModelToTest}:`, e.message); }

  try {
    console.log(`\n===== Llama.cpp Server (Model: ${llamaCppModelName}) =====`);
    const llamaCpp = LLMPlug.getProvider('llamacpp', { defaultModel: llamaCppModelName, baseURL: "http://localhost:8080/v1" });
    let llamaCppResult = await llamaCpp.chat(
        [{role: 'user', content: `Generate a 3-sentence summary of the plot of "The Matrix" using Llama.cpp server.`}],
        { maxTokens: 150 }
    );
    displayResult("Llama.cpp Server", "Chat", llamaCppResult);
    // Tool use highly dependent on model & server's OpenAI API feature completeness
    console.log("[Llama.cpp Server] Tool use support varies. Skipping advanced tool test for brevity.");
    await handleStream("Llama.cpp Server", "Chat Stream", llamaCpp.chatStream([{role: 'user', content: "What is gravity?"}], {maxTokens: 100}));
  } catch (e) { console.warn(`[Llama.cpp Server] Skipping model ${llamaCppModelName}:`, e.message); }

  try {
    console.log(`\n===== Oobabooga (Model: ${oobaModelName}) =====`);
    const oobabooga = LLMPlug.getProvider('oobabooga', { defaultModel: oobaModelName, baseURL: "http://localhost:5000/v1" });
    let oobaResult = await oobabooga.chat(
        [{role: 'user', content: `Write a short dialogue between a cat and a dog using Oobabooga.`}],
        { maxTokens: 200 }
    );
    displayResult("Oobabooga", "Chat", oobaResult);
    console.log("[Oobabooga] Tool use support varies. Skipping advanced tool test for brevity.");
    await handleStream("Oobabooga", "Chat Stream", oobabooga.chatStream([{role: 'user', content: "Why is the sky blue?"}], {maxTokens: 100}));
  } catch (e) { console.warn(`[Oobabooga] Skipping model ${oobaModelName}:`, e.message); }


  console.log("\nAll advanced examples finished.");
  rl.close();
}

main().catch(err => {
    console.error("\nUnhandled error in main execution:", err);
    rl.close();
});