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
  console.log("LLMPlug Usage Example\n");

  // --- Helper Functions for Displaying Results ---
  const displayResult = (providerName, type, result) => {
    console.log(`\n--- ${providerName} | ${type} ---`);
    if (result.text !== null && result.text !== undefined) console.log("Text:", result.text.trim());
    if (result.toolCalls && result.toolCalls.length > 0) {
      console.log("Tool Calls Requested:");
      result.toolCalls.forEach(tc => console.log(`  - ID: ${tc.id}, Function: ${tc.function.name}(${tc.function.arguments})`));
    }
    if (result.usage) {
        console.log("Usage:", `Prompt: ${result.usage.promptTokens || 'N/A'}, Completion: ${result.usage.completionTokens || 'N/A'}, Total: ${result.usage.totalTokens || 'N/A'}`);
    }
    if (result.finishReason) console.log("Finish Reason:", result.finishReason);
    console.log("---------------------------\n");
  };

  const handleStream = async (providerName, type, stream) => {
    console.log(`\n--- ${providerName} | ${type} (Streaming) ---`);
    let fullText = '';
    const toolCalls = {};
    let finalUsage;
    let finalFinishReason;

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
    console.log("---------------------------\n");
    return { text: fullText.trim(), toolCalls: Object.values(toolCalls), usage: finalUsage, finishReason: finalFinishReason };
  };

  // --- Tool Definitions and Execution Logic ---
  const tools = [
    {
      type: 'function',
      function: {
        name: 'get_current_weather',
        description: 'Get the current weather in a given location. Use this tool for any weather related queries.',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'The city and state, e.g. San Francisco, CA' },
            unit: { type: 'string', enum: ['celsius', 'fahrenheit'], description: 'Temperature unit' },
          },
          required: ['location'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'lookup_stock_price',
        description: 'Looks up the current price for a given stock ticker symbol. Use this tool for any stock price queries.',
        parameters: {
          type: 'object',
          properties: {
            ticker_symbol: { type: 'string', description: 'The stock ticker symbol, e.g., "AAPL" for Apple Inc.' },
          },
          required: ['ticker_symbol']
        },
      },
    }
  ];

  const availableTools = {
    get_current_weather: async ({ location, unit = 'fahrenheit' }) => {
      console.log(` MOCK TOOL: Calling 'get_current_weather' for ${location}, unit: ${unit}`);
      await new Promise(resolve => setTimeout(resolve, 500));
      if (location.toLowerCase().includes("san francisco")) {
        return { temperature: unit === 'celsius' ? '15' : '59', unit: unit, conditions: 'foggy' };
      }
      return { temperature: unit === 'celsius' ? '22' : '72', unit: unit, conditions: 'sunny' };
    },
    lookup_stock_price: async ({ ticker_symbol }) => {
        console.log(` MOCK TOOL: Calling 'lookup_stock_price' for ${ticker_symbol}`);
        await new Promise(resolve => setTimeout(resolve, 300));
        const prices = { "AAPL": 170.25, "GOOGL": 2750.50, "MSFT": 330.75 };
        const price = prices[ticker_symbol.toUpperCase()] || Math.floor(Math.random() * 1000) + 50;
        return { ticker: ticker_symbol.toUpperCase(), price: price, currency: "USD" };
    }
  };

  async function processToolCalls(provider, conversationHistory, toolCalls) {
    if (!toolCalls || toolCalls.length === 0) return conversationHistory;
    let updatedConversation = [...conversationHistory];
    updatedConversation.push({ role: 'assistant', content: null, tool_calls: toolCalls });

    for (const toolCall of toolCalls) {
      const functionName = toolCall.function.name;
      let functionArgs;
      try {
        functionArgs = JSON.parse(toolCall.function.arguments || '{}');
      } catch (e) {
        console.error(` MOCK TOOL: Error parsing arguments for ${functionName}: ${toolCall.function.arguments}`);
        updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: "Invalid arguments JSON", details: e.message } }] });
        continue;
      }
      if (availableTools[functionName]) {
        try {
          const toolOutputContent = await availableTools[functionName](functionArgs);
          console.log(` MOCK TOOL: Output for '${functionName}':`, toolOutputContent);
          updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: toolOutputContent }] });
        } catch (toolError) {
          console.error(` MOCK TOOL: Error executing tool '${functionName}':`, toolError.message);
          updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: toolError.message } }] });
        }
      } else {
        console.warn(` MOCK TOOL: Tool '${functionName}' not found.`);
         updatedConversation.push({ role: 'tool', tool_call_id: toolCall.id, name: functionName, content: [{ type: 'tool_output', tool_call_id: toolCall.id, content: { error: `Tool ${functionName} is not available.` } }] });
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
    let multimodalMessages = [{ role: 'user', content: [{ type: 'text', text: 'Describe this image:' }, { type: 'image_url', image_url: { url: catImageUrl, detail: 'low' } }]}];
    let openAIResult = await openai.chat(multimodalMessages, { maxTokens: 250 }); // Increased maxTokens
    displayResult("OpenAI", "Multimodal Chat (Cat Image)", openAIResult);

    let toolConversation = [{ role: 'user', content: "What's the weather in San Francisco, CA and what's the stock price for AAPL? Please use tools for both." }];
    for (let i = 0; i < 3; i++) {
        console.log(`OpenAI Tool Use - Iteration ${i + 1}`);
        const toolCallResult = await openai.chat(toolConversation, { tools: tools, maxTokens: 300 });
        displayResult("OpenAI", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            toolConversation = await processToolCalls(openai, toolConversation, toolCallResult.toolCalls);
        } else { console.log("OpenAI: No more tool calls requested."); break; }
        if (!toolCallResult.text && i < 2) { console.log("OpenAI: Model made tool calls, continuing..."); }
        else if (toolCallResult.text) { break; }
    }
    
    const jsonPrompt = "Return a JSON object with 'book_title' and 'author' for a fictional sci-fi novel.";
    openAIResult = await openai.chat(
      [{role: 'system', content: 'You are a helpful assistant that only responds with valid JSON. Do not include any other text.'}, { role: 'user', content: jsonPrompt }],
      { responseFormat: { type: 'json_object' }, maxTokens: 150 }
    );
    displayResult("OpenAI", "JSON Mode", openAIResult);
    if (openAIResult.text) try { console.log("Parsed JSON:", JSON.parse(openAIResult.text)); } catch (e) { console.error("Failed to parse JSON:", e); }

    const streamPrompt = "Tell me a very short, imaginative story about a star that learned to sing.";
    await handleStream("OpenAI", "Generate Stream (Story)", openai.generateStream(streamPrompt, { maxTokens: 200 }));

    console.log("OpenAI Streaming with Potential Tool Use (simplified handling):");
    const streamToolConversation = [{ role: 'user', content: 'What is the weather in London?' }];
    const streamedToolResponse = await handleStream("OpenAI", "Chat Stream (Tool)", openai.chatStream(streamToolConversation, { tools: tools, maxTokens: 200 }));
    if (streamedToolResponse.toolCalls && streamedToolResponse.toolCalls.length > 0) {
        console.log("OpenAI Stream: Tools were requested. Further processing would be needed.");
    }
  } catch (error) {
    console.error("OpenAI Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  // --- Anthropic Examples ---
  try {
    console.log("\n===== Anthropic =====");
    const anthropic = LLMPlug.getProvider('anthropic', { defaultModel: 'claude-3-haiku-20240307' });
    let anthropicResult = await anthropic.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'What is depicted in this logo?' }, { type: 'image_url', image_url: { url: logoImageUrl } }] }],
      { maxTokens: 200 } // Increased
    );
    displayResult("Anthropic", "Multimodal Chat (Logo Image)", anthropicResult);

    let anthropicToolConversation = [{ role: 'user', content: "Can you tell me the current weather in Paris? Please use a tool." }];
    for (let i = 0; i < 2; i++) {
        console.log(`Anthropic Tool Use - Iteration ${i + 1}`);
        const toolCallResult = await anthropic.chat(anthropicToolConversation, { tools: tools, maxTokens: 300 });
        displayResult("Anthropic", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            anthropicToolConversation = await processToolCalls(anthropic, anthropicToolConversation, toolCallResult.toolCalls);
        } else { console.log("Anthropic: No more tool calls requested."); break; }
        if (toolCallResult.text) break;
    }
    await handleStream("Anthropic", "Generate Stream (Explanation)", anthropic.generateStream("Explain the concept of a black hole in simple terms.", { maxTokens: 250 })); // Increased
  } catch (error) {
    console.error("Anthropic Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  // --- Google Gemini Examples ---
  try {
    console.log("\n===== Google Gemini =====");
    const google = LLMPlug.getProvider('google', { defaultModel: 'gemini-2.0-pro-exp-02-05' });
    let geminiResult = await google.chat(
      [{ role: 'user', content: [{ type: 'text', text: 'What animal is this and what might it be thinking?' }, { type: 'image_url', image_url: { url: catImageUrl } }] }],
      { maxTokens: 300 } // Increased
    );
    displayResult("Google Gemini", "Multimodal Chat (Cat Image)", geminiResult);

    let geminiToolConversation = [
        {role: 'system', content: 'You are a helpful assistant. When a user asks for information that can be retrieved by a tool, you MUST use the appropriate tool. Do not apologize or say you cannot do something if a tool exists for it.'},
        { role: 'user', content: "Use your tools to find the current stock price for MSFT." }
    ];
     for (let i = 0; i < 2; i++) {
        console.log(`Gemini Tool Use - Iteration ${i + 1}`);
        const toolCallResult = await google.chat(geminiToolConversation, { tools: tools, maxTokens: 250 });
        displayResult("Google Gemini", `Tool Call Step ${i + 1}`, toolCallResult);
        if (toolCallResult.toolCalls && toolCallResult.toolCalls.length > 0) {
            geminiToolConversation = await processToolCalls(google, geminiToolConversation, toolCallResult.toolCalls);
        } else { console.log("Gemini: No more tool calls requested."); break; }
        if (toolCallResult.text) break;
    }

    await handleStream("Google Gemini", "Generate Stream (Poem)", google.generateStream("Write a short, optimistic poem about the future of AI.", { maxTokens: 200 })); // Increased

    const geminiJsonMessages = [
        { role: 'system', content: "You are an API. Your SOLE function is to return a valid JSON object. Do NOT include any introductory text, explanations, apologies, or conversational filler. Your entire response must be a single, parsable JSON object and nothing else."},
        { role: 'user', content: "Provide a JSON object detailing a planet: name (string), type (e.g., 'Gas Giant', 'Terrestrial'), and moons (number)." }
    ];
    geminiResult = await google.chat(
      geminiJsonMessages,
      { responseFormat: { type: 'json_object' }, maxTokens: 200 } // Increased
    );
    displayResult("Google Gemini", "JSON Mode", geminiResult);
    if (geminiResult.text) {
        try {
            console.log("Parsed JSON:", JSON.parse(geminiResult.text));
        } catch (e) {
            console.warn("Gemini JSON Mode: Failed to parse JSON directly, attempting extraction...");
            const jsonMatch = geminiResult.text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
            if (jsonMatch && jsonMatch[0]) {
                try {
                    console.log("Extracted and Parsed JSON:", JSON.parse(jsonMatch[0]));
                } catch (e2) {
                    console.error("Failed to parse extracted JSON:", e2, "\nOriginal text:", geminiResult.text);
                }
            } else {
                console.error("No JSON object or array found in the response string:", geminiResult.text);
            }
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
    const hfResult = await hf.generate("What is the main benefit of using a large language model?", { maxTokens: 100 }); // Increased
    displayResult("Hugging Face", "Generate", hfResult);
    const hfChatResult = await hf.chat(
      [{ role: 'user', content: 'What is the capital of Canada?' }, { role: 'assistant', content: 'The capital of Canada is Ottawa.' }, { role: 'user', content: 'And what is its largest city?' }], 
      { maxTokens: 70 } // Increased
    );
    displayResult("Hugging Face", "Chat", hfChatResult);
    console.log("Hugging Face: Attempting streaming (expected to fail)...");
    try {
        await handleStream("HuggingFace", "Generate Stream (Error Expected)", hf.generateStream("This will fail."));
    } catch (e) {
        console.log("Hugging Face Streaming Error (As Expected):", e.message);
    }
  } catch (error) {
    console.error("Hugging Face Error:", error.message);
    if (error instanceof LLMPlugError && error.originalError) console.error("Original Error:", error.originalError.toString());
  }

  console.log("\nAll examples finished.");
  rl.close();
}

main().catch(err => {
    console.error("\nUnhandled error in main execution:", err);
    rl.close();
});