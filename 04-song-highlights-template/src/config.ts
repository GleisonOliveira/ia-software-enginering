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

  memory: {
    dbUri: string;
  };
};

console.assert(
  process.env.API_KEY,
  "OPENROUTER_API_KEY is not set in environment variables",
);
console.assert(
  process.env.BASE_URL,
  "BASE_URL is not set in environment variables",
);

export const config: ModelConfig = {
  apiKey: process.env.API_KEY!,
  baseURL: process.env.BASE_URL!,
  httpReferer: "",
  xTitle: "IA Devs - Prompt Chaining Article Generator",
  models: [
    // 'qwen/qwen3-coder-next',
    // https://openrouter.ai/models?fmt=cards&max_price=0&order=throughput-high-to-low&supported_parameters=structured_outputs%2Cresponse_format
    // "openprovider/auto-free",
    "openrouter/owl-alpha",
    // 'gpt-oss-120b:free',
  ],
  provider: {
    sort: {
      by: "throughput", // Route to model with highest throughput (fastest response)
      partition: "none",
    },
  },
  temperature: 0.7,
  memory: {
    dbUri:
      "postgresql://postgres:mysecretpassword@localhost:5433/song_recommender",
  },
};
