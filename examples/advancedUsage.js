import { LLMPlug, LLMPlugError, LLMPlugToolError } from '../src/index.js';
// For Node.js < 20.6.0 or if not using --env-file, you might need to load .env manually:
// import dotenv from 'dotenv';
// dotenv.config();

import readline from 'node:readline/promises';
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function main() {
  console.log("LLMPlug Advanced Usage Showcase\n");

  // --- Helper Functions for Displaying Results ---
  const displayResult = (providerName, type, result, note = "") => {
    console.log(`\n--- ${providerName} | ${type} ${note} ---`);
    if (result.text !== null && result.text !== undefined) console.log("Text:", result.text.trim());
    if (result.toolCalls && result.toolCalls.length > 0) {
      console.log("Tool Calls Requested:");
      result.toolCalls.forEach(tc => console.log(`  - ID: ${tc.id}, Function: ${tc.function.name}(${tc.function.arguments})`));
    }
    if (result.usage) {
        console.log("Usage:", `Prompt: ${result.usage.promptTokens || 'N/A'}, Completion: ${result.usage.completionTokens || 'N/A'}, Total: ${result.usage.totalTokens || 'N/A'}`);
    }
    if (result.finishReason) console.log("Finish Reason:", result.finishReason);
    if (result.rawResponse?.candidates?.[0]?.safetyRatings) { // For Gemini
        console.log("Safety Ratings:", JSON.stringify(result.rawResponse.candidates[0].safetyRatings));
    }
    console.log("-----------------------------------\n");
  };

  const handleStream = async (providerName, type, stream) => {
    console.log(`\n--- ${providerName} | ${type} (Streaming) ---`);
    let fullText = '';
    const toolCalls = {};
    let finalUsage;
    let finalFinishReason;
    let lastSafetyRatings;

    process.stdout.write("Streamed Text: ");
    for await (const chunk of stream) {
      if (chunk.text) {
        process.stdout.write(chunk.text);
        fullText += chunk.text;
      }
      if (chunk.toolCalls) {
        chunk.toolCalls.forEach(tcChunk => {
            if (!toolCalls[tcChunk.id]) {
                toolCalls[tcChunk.id] = { ...tcChunk, function: { name: tcChunk.function.name, arguments: '' } };
            }
            if (tcChunk.function.arguments) {
                toolCalls[tcChunk.id].function.arguments += tcChunk.function.arguments;
            }
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
    if (finalUsage) {
        console.log("Stream Usage:", `Prompt: ${finalUsage.promptTokens || 'N/A'}, Completion: ${finalUsage.completionTokens || 'N/A'}, Total: ${finalUsage.totalTokens || 'N/A'}`);
    }
    if (finalFinishReason) console.log("Stream Finish Reason:", finalFinishReason);
    if (lastSafetyRatings) console.log("Stream Safety Ratings (last chunk):", JSON.stringify(lastSafetyRatings));
    console.log("-----------------------------------\n");
    return { text: fullText.trim(), toolCalls: Object.values(toolCalls), usage: finalUsage, finishReason: finalFinishReason };
  };

  // --- Tool Definitions and Execution Logic ---
  const tools = [
    { type: 'function', function: { name: 'get_current_weather', description: 'Get the current weather. Use for weather queries.', parameters: { type: 'object', properties: { location: { type: 'string', description: 'City and state, e.g. San Francisco, CA' }, unit: { type: 'string', enum: ['celsius', 'fahrenheit'] } }, required: ['location'] } } },
    { type: 'function', function: { name: 'lookup_stock_price', description: 'Get current stock price. Use for stock queries.', parameters: { type: 'object', properties: { ticker_symbol: { type: 'string', description: 'Stock ticker, e.g., AAPL' } }, required: ['ticker_symbol'] } } },
    { type: 'function', function: { name: 'generate_python_code', description: 'Generates executable Python code based on a description. Returns the code as a string.', parameters: { type: 'object', properties: { task_description: { type: 'string', description: 'A detailed description of the Python task to perform.' } }, required: ['task_description']}}}
  ];

  const availableTools = {
    get_current_weather: async ({ location, unit = 'fahrenheit' }) => {
      console.log(` MOCK TOOL: 'get_current_weather' for ${location}, ${unit}`);
      await new Promise(resolve => setTimeout(resolve, 200));
      return { temperature: location.toLowerCase().includes("san francisco") ? '15C/59F' : '22C/72F', conditions: 'variable' };
    },
    lookup_stock_price: async ({ ticker_symbol }) => {
        console.log(` MOCK TOOL: 'lookup_stock_price' for ${ticker_symbol}`);
        await new Promise(resolve => setTimeout(resolve, 150));
        return { ticker: ticker_symbol.toUpperCase(), price: (Math.random() * 500 + 50).toFixed(2), currency: "USD" };
    },
    generate_python_code: async ({ task_description }) => {
        console.log(` MOCK TOOL: 'generate_python_code' for task: "${task_description}"`);
        await new Promise(resolve => setTimeout(resolve, 100));
        // This is a mock. A real tool might use a sandboxed execution environment or another LLM.
        return `print("Python code for: ${task_description.replace(/"/g, '\\"')}")\n# Placeholder code\npass`;
    }
  };

  async function processToolCalls(provider, conversationHistory, toolCalls) {
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

  // --- OpenAI Examples ---
  try {
    console.log("===== OpenAI =====");
    const openai = LLMPlug.getProvider('openai', { defaultModel: 'gpt-4o' });
    let openAIResult = await openai.chat(
        [{ role: 'user', content: [{ type: 'text', text: 'Describe this cat image:' }, { type: 'image_url', image_url: { url: catImageUrl, detail: 'auto' } }]}],
        { maxTokens: 250 }
    );
    displayResult("OpenAI", "Multimodal Chat", openAIResult);

    let toolConversation = [{role: 'system', content: 'You are a helpful assistant that uses tools effectively to answer user queries. Ask for clarification if needed before using a tool.'}, { role: 'user', content: "What's the weather in Tokyo and the stock price for GOOGL? Then, write a python function to calculate factorial." }];
    for (let i = 0; i < 4; i++) { // Increased iterations for multi-tool scenario
        console.log(`OpenAI Tool Use - Iteration ${i + 1}`);
        const toolCallResult = await openai.chat(toolConversation, { tools: tools, maxTokens: 400 });
        displayResult("OpenAI", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            toolConversation = await processToolCalls(openai, toolConversation, toolCallResult.toolCalls);
        } else { console.log("OpenAI: No more tool calls."); break; }
        if (toolCallResult.text && i > 0) { break; } // Stop if model gives text after first iteration
    }
    
    openAIResult = await openai.chat(
      [{role: 'system', content: 'Respond ONLY with valid JSON.'}, { role: 'user', content: "Create a JSON object for a user profile: name, email, and isActive (boolean)." }],
      { responseFormat: { type: 'json_object' }, maxTokens: 2500 }
    );
    displayResult("OpenAI", "JSON Mode", openAIResult);
    if (openAIResult.text) try { console.log("Parsed JSON:", JSON.parse(openAIResult.text)); } catch (e) { console.error("Failed to parse JSON:", e); }

    await handleStream("OpenAI", "Generate Stream (Poem)", openai.generateStream("Compose a short, witty poem about debugging code.", { maxTokens: 2500 }));
  } catch (error) {
    console.error("OpenAI Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  // --- Anthropic Examples ---
  try {
    console.log("\n===== Anthropic =====");
    const anthropic = LLMPlug.getProvider('anthropic', { defaultModel: 'claude-3-sonnet-20240229' }); // Sonnet is more capable
    let anthropicResult = await anthropic.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'Explain this logo and its purpose:' }, { type: 'image_url', image_url: { url: logoImageUrl } }] }],
      { maxTokens: 2500 }
    );
    displayResult("Anthropic", "Multimodal Chat", anthropicResult);

    let anthropicToolConversation = [{ role: 'user', content: "What's the weather in Berlin, Germany? Then use another tool to generate python code for a function that reverses a string." }];
    for (let i = 0; i < 3; i++) {
        console.log(`Anthropic Tool Use - Iteration ${i + 1}`);
        const toolCallResult = await anthropic.chat(anthropicToolConversation, { tools: tools, maxTokens: 400 });
        displayResult("Anthropic", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            anthropicToolConversation = await processToolCalls(anthropic, anthropicToolConversation, toolCallResult.toolCalls);
        } else { console.log("Anthropic: No more tool calls."); break; }
        if (toolCallResult.text && i > 0) break;
    }
    await handleStream("Anthropic", "Generate Stream (Story)", anthropic.generateStream("Write a brief science fiction story premise involving a newly discovered exoplanet.", { maxTokens: 250 }));
  } catch (error) {
    console.error("Anthropic Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  // --- Google Gemini Examples ---
  try {
    console.log("\n===== Google Gemini =====");
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-2.0-pro-exp-02-05' });
    let geminiResult = await google.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'Identify this animal and suggest a funny caption for this image.' }, { type: 'image_url', image_url: { url: catImageUrl } }] }],
      { maxTokens: 2500 }
    );
    displayResult("Google Gemini", "Multimodal Chat", geminiResult);

    let geminiToolConversation = [
        {role: 'system', content: 'You are a powerful assistant. You MUST use available tools to answer queries accurately. If a query requires code generation, use the "generate_python_code" tool.'},
        { role: 'user', content: "What's the weather in Rome? Also, please generate Python code for a function that checks if a number is prime." }
    ];
    let geminiSafetyRetry = false;
     for (let i = 0; i < 3; i++) {
        console.log(`Gemini Tool Use - Iteration ${i + 1}${geminiSafetyRetry ? ' (Permissive Safety)' : ''}`);
        const toolCallOptions = { 
            tools: tools, 
            maxTokens: 2500,
            extraParams: { usePermissiveSafety: geminiSafetyRetry } // Apply permissive safety if retrying
        };
        const toolCallResult = await google.chat(geminiToolConversation, toolCallOptions);
        displayResult("Google Gemini", `Tool Call Step ${i + 1}`, toolCallResult, geminiSafetyRetry ? '(Permissive Safety)' : '');

        if (toolCallResult.text === null && toolCallResult.finishReason === 'SAFETY' && !geminiSafetyRetry) {
            console.warn("Gemini: Tool call or response blocked by safety. Retrying with permissive safety ONCE.");
            geminiSafetyRetry = true; // Set flag to retry with permissive settings
            i--; // Decrement i to retry the current iteration
            continue; // Skip to next iteration which will now use permissive settings
        }
        geminiSafetyRetry = false; // Reset flag after a successful attempt or non-safety failure

        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            geminiToolConversation = await processToolCalls(google, geminiToolConversation, toolCallResult.toolCalls);
        } else { console.log("Gemini: No more tool calls."); break; }
        if (toolCallResult.text && i > 0) break;
    }

    await handleStream("Google Gemini", "Generate Stream (Tech Explanation)", google.generateStream("Explain how a transformer model works in simple terms for a beginner.", { maxTokens: 300 }));
    
    const geminiJsonMessages = [
        { role: 'system', content: "You are an API. Your ONLY response is a single, valid JSON object. No other text or explanation."},
        { role: 'user', content: "Generate JSON for a product: id (number), name (string), price (number), inStock (boolean)." }
    ];
    geminiResult = await google.chat(geminiJsonMessages, { responseFormat: { type: 'json_object' }, maxTokens: 200 });
    displayResult("Google Gemini", "JSON Mode", geminiResult);
    if (geminiResult.text) {
        try { console.log("Parsed JSON:", JSON.parse(geminiResult.text)); } 
        catch (e) {
            const jsonMatch = geminiResult.text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
            if (jsonMatch && jsonMatch[0]) try { console.log("Extracted & Parsed JSON:", JSON.parse(jsonMatch[0])); } 
            catch (e2) { console.error("Failed to parse extracted JSON:", e2, "\nOriginal:", geminiResult.text); }
            else { console.error("No JSON found in response:", geminiResult.text); }
        }
    }
  } catch (error) {
    console.error("Google Gemini Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  // --- Hugging Face Example (Basic Functionality) ---
  try {
    console.log("\n===== Hugging Face =====");
    const hf = LLMPlug.getProvider('huggingface', { modelId: 'mistralai/Mistral-7B-Instruct-v0.1' }); 
    const hfResult = await hf.generate("Explain the concept of 'machine learning bias' in two sentences.", { maxTokens: 120 });
    displayResult("Hugging Face", "Generate", hfResult);
    console.log("Hugging Face: Advanced features like tool use, streaming, JSON mode are generally not supported via generic Inference API.");
  } catch (error) {
    console.error("Hugging Face Error:", error.message);
  }

  console.log("\nAll advanced examples finished.");
  rl.close();
}

main().catch(err => {
    console.error("\nUnhandled error in main execution:", err);
    rl.close();
});