export type ModelConfig = {
  apiKey: string;
  baseURL: string;
  httpReferer: string;
  xTitle: string;

  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };

  models: string[];
  temperature: number;
  maxTokens: number;
};

console.assert(
  process.env.API_KEY,
  "API_KEY is not set in environment variables",
);

export const config: ModelConfig = {
  apiKey: process.env.API_KEY!,
  baseURL: process.env.BASE_URL!,
  httpReferer: "",
  xTitle: "IA Devs - Transforming Services into Tools",
  models: ["google/gemma-4-26b-a4b-it:free"],
  provider: {
    sort: {
      by: "throughput", // Route to model with highest throughput (fastest response)
      partition: "none",
    },
  },
  temperature: 0.7,
  maxTokens: 2048,
};
