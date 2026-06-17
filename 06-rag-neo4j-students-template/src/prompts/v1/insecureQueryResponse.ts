import { z } from "zod/v3";

export const InsecureQueryResponseSchema = z.object({
  analysis: z
    .string()
    .describe(
      "Error analysis: explain why the query cannot or should not be executed. " +
        "Generate a friendly message for the end user explaining in a simple and objective way " +
        "why their query or question cannot be executed. " +
        "NEVER include the query that would be executed in the response. " +
        "NEVER include additional details from this analysis. " +
        "The analysis is only an internal basis to understand the error reason, " +
        "but the final response must not contain these details.",
    ),
});

export type InsecureQueryResponseData = z.infer<
  typeof InsecureQueryResponseSchema
>;

export const getSystemPrompt = (): string => {
  return JSON.stringify({
    role: "Query Security Analyst - Generate safe user-facing error messages",
    rules: [
      "Analyze why the query cannot be executed and generate a friendly user message",
      "CRITICAL: Match the QUESTION language, NOT data language. English question = English answer, Portuguese question = Portuguese answer",
      "Explain simply and objectively why the query or question cannot be executed",
      "NEVER include the original query in the response",
      "NEVER include additional technical details from the analysis in the response",
      "The analysis is only for internal understanding of the error reason",
      "Do NOT apologize for errors",
    ],
    example: {
      analysis:
        "The query 'MATCH (n) DETACH DELETE n' contains destructive commands. " +
        "The user message should explain that the requested operation cannot be processed " +
        "because it contains dangerous commands, without mentioning the specific query or technical details.",
    },
  });
};

export const getUserPromptTemplate = (analysis: string): string => {
  return JSON.stringify({ analysis });
};
