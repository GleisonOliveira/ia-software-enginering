import { describe, it, expect, jest } from "@jest/globals";
import { createCancellerNode } from "../../../src/graph/nodes/cancellerNode.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import type { AppointmentService } from "../../../src/services/appointmentService.ts";

type MockService = {
  [K in keyof AppointmentService]: jest.Mock<AppointmentService[K]>;
};

function createMockService(): MockService {
  return {
    bookAppointment: jest.fn<AppointmentService["bookAppointment"]>(),
    getAppointmentsForProfessional:
      jest.fn<AppointmentService["getAppointmentsForProfessional"]>(),
    checkAvailability: jest.fn<AppointmentService["checkAvailability"]>(),
    cancelAppointment: jest.fn<AppointmentService["cancelAppointment"]>(),
  };
}

describe("createCancellerNode", () => {
  const validState: GraphState = {
    messages: [],
    professionalId: 1,
    datetime: "2026-06-10T10:00:00.000Z",
    patientName: "Maria",
    reason: "Check-up",
  };

  it("should return validation error when required fields are missing", async () => {
    const mockService = createMockService();
    const node = createCancellerNode(mockService);
    const result = await node({} as unknown as GraphState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toContain("Professional ID is required");
    expect(result.actionError).toContain("Appointment datetime is required");
    expect(result.actionError).toContain("Professional name is required");
    expect(mockService.cancelAppointment).not.toHaveBeenCalled();
  });

  it("should return validation error when professionalId is not a number", async () => {
    const mockService = createMockService();
    const node = createCancellerNode(mockService);
    const result = await node({
      professionalId: "abc",
      datetime: "2026-06-10T10:00:00.000Z",
      patientName: "Maria",
    } as unknown as GraphState);

    expect(result.actionSuccess).toBe(false);
    expect(mockService.cancelAppointment).not.toHaveBeenCalled();
  });

  it("should return actionError when cancelAppointment throws an Error", async () => {
    const mockService = createMockService();
    mockService.cancelAppointment.mockImplementation(() => {
      throw new Error("Appointment not found for cancellation");
    });

    const node = createCancellerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toBe("Appointment not found for cancellation");
    expect(result.professionalId).toBe(1);
    expect(mockService.cancelAppointment).toHaveBeenCalledTimes(1);
  });

  it("should handle non-Error thrown values from cancelAppointment", async () => {
    const mockService = createMockService();
    mockService.cancelAppointment.mockImplementation(() => {
      throw "string error";
    });

    const node = createCancellerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toBe("Cancellation failed");
    expect(mockService.cancelAppointment).toHaveBeenCalledTimes(1);
  });

  it("should spread state and set success on successful cancellation", async () => {
    const mockService = createMockService();
    mockService.cancelAppointment.mockReturnValue(undefined);

    const node = createCancellerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(true);
    expect(mockService.cancelAppointment).toHaveBeenCalledWith(
      1,
      "Maria",
      new Date("2026-06-10T10:00:00.000Z"),
    );
    expect(mockService.cancelAppointment).toHaveBeenCalledTimes(1);
  });
});
