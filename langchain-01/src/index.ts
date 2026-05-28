import Fastify from "fastify";
import { buildGraph } from "./graph/graph";
import { HumanMessage } from "langchain";

const app = Fastify({ logger: true });
const graph = buildGraph();

const chatSchema = {
  body: {
    type: "object",
    required: ["question"],
    properties: {
      question: { type: "string", minLength: 3 },
    },
  },
};

app.post("/chat", { schema: chatSchema }, async (request, reply) => {
  const { question } = request.body as { question: string };
  const response = await graph.invoke({
    messages: [new HumanMessage(question)],
  });

  return reply.send({ response: response.output });
});

app.listen({ port: 3000, host: "0.0.0.0" }, (err) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
});
