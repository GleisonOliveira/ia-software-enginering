import { describe, it, expect, jest } from "@jest/globals";
import { createSchedulerNode } from "../../../src/graph/nodes/schedulerNode.ts";
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

describe("createSchedulerNode", () => {
  const validState: GraphState = {
    messages: [],
    professionalId: 1,
    datetime: "2026-06-10T10:00:00.000Z",
    patientName: "Maria",
    reason: "Check-up",
  };

  const appointmentData = {
    date: "2026-06-10T10:00:00.000Z",
    patientName: "Maria",
    reason: "Check-up",
    professionalId: 1,
  };

  it("should return validation error when required fields are missing", async () => {
    const mockService = createMockService();
    const node = createSchedulerNode(mockService);
    const result = await node({} as unknown as GraphState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toContain("Professional ID is required");
    expect(result.actionError).toContain("Appointment datetime is required");
    expect(result.actionError).toContain("Professional name is required");
    expect(mockService.bookAppointment).not.toHaveBeenCalled();
  });

  it("should return validation error when professionalId is not a number", async () => {
    const mockService = createMockService();
    const node = createSchedulerNode(mockService);
    const result = await node({
      professionalId: "abc",
      datetime: "2026-06-10T10:00:00.000Z",
      patientName: "Maria",
    } as unknown as GraphState);

    expect(result.actionSuccess).toBe(false);
    expect(mockService.bookAppointment).not.toHaveBeenCalled();
  });

  it("should return actionError when bookAppointment throws an Error", async () => {
    const mockService = createMockService();
    mockService.bookAppointment.mockImplementation(() => {
      throw new Error("Horário indisponível para este profissional");
    });

    const node = createSchedulerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toBe(
      "Horário indisponível para este profissional",
    );
    expect(result.professionalId).toBe(1);
    expect(mockService.bookAppointment).toHaveBeenCalledTimes(1);
  });

  it("should handle non-Error thrown values from bookAppointment", async () => {
    const mockService = createMockService();
    mockService.bookAppointment.mockImplementation(() => {
      throw "string error";
    });

    const node = createSchedulerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(false);
    expect(result.actionError).toBe("Scheduling failed");
    expect(mockService.bookAppointment).toHaveBeenCalledTimes(1);
  });

  it("should spread state and set success on successful booking", async () => {
    const mockService = createMockService();
    mockService.bookAppointment.mockReturnValue(appointmentData);

    const node = createSchedulerNode(mockService);
    const result = await node(validState);

    expect(result.actionSuccess).toBe(true);
    expect(result.appointmentData).toEqual(appointmentData);
    expect(mockService.bookAppointment).toHaveBeenCalledWith(
      1,
      new Date("2026-06-10T10:00:00.000Z"),
      "Maria",
      "Check-up",
    );
    expect(mockService.bookAppointment).toHaveBeenCalledTimes(1);
  });
});
