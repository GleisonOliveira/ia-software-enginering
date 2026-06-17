import { AIMessage } from "langchain";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  AnalyticalResponseSchema,
  getErrorResponsePrompt,
  getMultiStepSynthesisPrompt,
  getNoResultsPrompt,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/analyticalResponse.ts";

async function handleErrorResponse(
  { error, question }: GraphState,
  llmClient: OpenRouterService,
): Promise<Partial<GraphState>> {
  const systemPrompt = getSystemPrompt();
  const userUserPrompt = getErrorResponsePrompt(error!, question);
  const result = await llmClient.generateStructured(
    systemPrompt,
    userUserPrompt,
    AnalyticalResponseSchema,
  );

  if (!result.success) {
    const { error } = result;

    return {
      messages: [new AIMessage(`An error ocurred: ${error}`)],
      error,
      answer: `An error ocurred: ${error}`,
      followUpQuestions: [],
    };
  }

  const { answer, followUpQuestions } = result.data;

  return {
    messages: [new AIMessage(answer!)],
    answer: answer,
    followUpQuestions: followUpQuestions,
  };
}

async function handleSuccessResponse(
  {
    isMultiStep,
    subResults,
    subQuestions,
    subQueries,
    question,
    query,
    dbResults,
  }: GraphState,
  llmClient: OpenRouterService,
): Promise<Partial<GraphState>> {
  const systemPrompt = getSystemPrompt();
  let _userPrompt: string;

  if (
    Boolean(
      isMultiStep &&
      subResults?.length &&
      subQuestions?.length &&
      subQueries?.length,
    )
  ) {
    const stepsData = subResults!.map((results, index) => ({
      stepNumber: index + 1,
      question: subQuestions![index],
      query: subQueries![index],
      results: JSON.stringify(results),
    }));

    _userPrompt = getMultiStepSynthesisPrompt(question!, stepsData);
  } else {
    _userPrompt = getUserPromptTemplate(
      question!,
      query!,
      JSON.stringify(dbResults),
    );
  }

  const result = await llmClient.generateStructured(
    systemPrompt,
    _userPrompt,
    AnalyticalResponseSchema,
  );

  if (!result.success) {
    return {
      error: `Reponse generation faild: ${result.error ?? "Unknown error"}`,
    };
  }

  const { answer, followUpQuestions } = result.data;

  return {
    messages: [new AIMessage(answer!)],
    answer: answer,
    followUpQuestions: followUpQuestions,
  };
}

async function handleNoResultsResponse(
  state: GraphState,
  llmClient: OpenRouterService,
): Promise<GraphState> {
  const systemPrompt = getSystemPrompt();
  const userPrompt = getNoResultsPrompt(
    state.question ?? "your query",
    state.query ?? "N/A",
  );

  const result = await llmClient.generateStructured(
    userPrompt,
    systemPrompt,
    AnalyticalResponseSchema,
  );

  if (result.success) {
    const { answer, followUpQuestions } = result.data;

    return {
      ...state,
      messages: [...state.messages, new AIMessage(answer)],
      answer: answer,
      followUpQuestions: followUpQuestions,
    };
  }

  const noResultsMessage = "No data found matching your query.";

  return {
    ...state,
    messages: [...state.messages, new AIMessage(noResultsMessage)],
    error: result.error,
    answer: noResultsMessage,
    followUpQuestions: [],
  };
}

export function createAnalyticalResponseNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      if (state.error) {
        return await handleErrorResponse(state, llmClient);
      }

      if (!state.dbResults?.length) {
        return await handleNoResultsResponse(state, llmClient);
      }

      return await handleSuccessResponse(state, llmClient);
    } catch (error: any) {
      return {
        ...state,
        error: `Response generation failed: ${error.message}`,
      };
    }
  };
}
