import { afterAll, afterEach, describe, expect, it, jest } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const home = mkdtempSync(join(tmpdir(), "opencode-mem-idle-"));
const originalHome = process.env.HOME;
const originalUserProfile = process.env.USERPROFILE;
process.env.HOME = home;
process.env.USERPROFILE = home;

const { CONFIG } = await import("../src/config.js");
const { EmbeddingService } = await import("../src/services/embedding.js");

afterAll(() => {
  process.env.HOME = originalHome;
  process.env.USERPROFILE = originalUserProfile;
});

afterEach(() => jest.useRealTimers());

function makeService(): { service: any; disposed: string[] } {
  const disposed: string[] = [];
  const service = new EmbeddingService();
  service.local = {
    run: async () => ({ type: "vector", vector: new Float32Array([1, 2, 3]) }),
    close: async () => {
      disposed.push("dispose");
    },
  };
  service.isWarmedUp = true;
  return { service, disposed };
}

async function withIdleTimeout(ms: number, run: () => Promise<void>): Promise<void> {
  const previous = CONFIG.embeddingIdleTimeoutMs;
  const previousUrl = CONFIG.embeddingApiUrl;
  const previousKey = CONFIG.embeddingApiKey;
  CONFIG.embeddingIdleTimeoutMs = ms;
  CONFIG.embeddingApiUrl = undefined;
  CONFIG.embeddingApiKey = undefined;
  try {
    await run();
  } finally {
    CONFIG.embeddingIdleTimeoutMs = previous;
    CONFIG.embeddingApiUrl = previousUrl;
    CONFIG.embeddingApiKey = previousKey;
  }
}

describe("EmbeddingService idle release", () => {
  it("disposes the local pipeline after the idle timeout", async () => {
    await withIdleTimeout(20, async () => {
      jest.useFakeTimers();
      const { service, disposed } = makeService();
      await service.embed("hello");
      expect(disposed).toEqual([]);

      jest.advanceTimersByTime(20);

      expect(disposed).toEqual(["dispose"]);
      expect(service.isWarmedUp).toBe(false);
      expect(service.local).toBeNull();
    });
  });

  it("keeps the model loaded when the timeout is 0", async () => {
    await withIdleTimeout(0, async () => {
      jest.useFakeTimers();
      const { service, disposed } = makeService();
      await service.embed("hello");

      jest.advanceTimersByTime(80);

      expect(disposed).toEqual([]);
      expect(service.isWarmedUp).toBe(true);
    });
  });

  it("defers disposal while an embed is in flight and releases it afterwards", async () => {
    await withIdleTimeout(0, async () => {
      const { service, disposed } = makeService();
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      service.local = {
        run: async () => {
          await gate;
          return { type: "vector", vector: new Float32Array([1]) };
        },
        close: async () => disposed.push("dispose"),
      };

      const running = service.embed("in-flight");
      await service.dispose("client-close");

      // The running embed still holds the pipeline, so nothing is released yet.
      expect(disposed).toEqual([]);

      release();
      await running;

      // Release happens without a second dispose() call, even with the timer off.
      expect(disposed).toEqual(["dispose"]);
      expect(service.isWarmedUp).toBe(false);
      expect(service.local).toBeNull();
    });
  });
});
