import { config } from "../config.js";
import { AppointmentService } from "../services/appointmentService.js";
import { OpenRouterService } from "../services/openRouterService.js";
import { buildAppointmentGraph } from "./graph.js";

export function buildGraph() {
  const llmClient = new OpenRouterService(config);
  const appointmentService = new AppointmentService();

  return buildAppointmentGraph(llmClient, appointmentService);
}

export const graph = async () => {
  return buildGraph();
};

export default graph;
