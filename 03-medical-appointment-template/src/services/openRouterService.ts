import { ChatOpenAI } from "@langchain/openai";
import { config, ModelConfig } from "../config.js";
import { z } from "zod/v3";
import {
  createAgent,
  HumanMessage,
  providerStrategy,
  SystemMessage,
} from "langchain";
import { success } from "zod/v4";

export class OpenRouterService {
  private config: ModelConfig;
  private llmClient: ChatOpenAI;

  constructor(configOvveride?: ModelConfig) {
    this.config = configOvveride ?? config;
    this.llmClient = new ChatOpenAI({
      apiKey: this.config.apiKey,
      model: this.config.models[0],
      temperature: this.config.temperature,
      configuration: {
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
          "HTTP-Referer": this.config.httpReferer,
          "X-Title": this.config.xTitle,
        },
      },
      modelKwargs: {
        models: this.config.models,
        provider: this.config.provider,
      },
    });
  }

  async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema: z.ZodType<T>,
  ) {
    // agent IA connection
    const agent = createAgent({
      model: this.llmClient,
      tools: [],
      responseFormat: providerStrategy(schema),
    });

    const messages = [
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ];

    try {
      const data = await agent.invoke({ messages });
      1;

      return {
        success: true,
        data: data.structuredResponse,
      };
    } catch (error) {
      console.error(error);

      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}
