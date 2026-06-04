import type { AppointmentService } from "../../services/appointmentService.ts";
import type { GraphState } from "../graph.ts";
import { z } from "zod/v3";

const scheduleRequiredFieldsSchema = z.object({
  professionalId: z.number({ required_error: "Professional ID is required" }),
  datetime: z.string({ required_error: "Appointment datetime is required" }),
  patientName: z.string({ required_error: "Professional name is required" }),
});

export function createSchedulerNode(appointmentService: AppointmentService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const validation = scheduleRequiredFieldsSchema.safeParse(state);

      if (!validation.success) {
        const errorMessages = validation.error.errors
          .map((e) => e.message)
          .join(",");

        return {
          actionSuccess: false,
          actionError: errorMessages,
        };
      }

      const { professionalId, datetime, patientName } = validation.data;
      const appointment = appointmentService.bookAppointment(
        professionalId,
        new Date(datetime),
        patientName,
        state.reason ?? "",
      );

      return {
        actionSuccess: true,
        appointmentData: appointment,
      };
    } catch (error) {
      return {
        ...state,
        actionSuccess: false,
        actionError:
          error instanceof Error ? error.message : "Scheduling failed",
      };
    }
  };
}
