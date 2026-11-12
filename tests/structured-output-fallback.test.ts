import { describe, expect, it } from "bun:test";
import { randomUUID } from "node:crypto";
import type { Context } from "@opencode/plugin/promise/plugin";
import { z } from "zod";
import { createLegacyClient } from "../src/v2/legacy-client.js";

const primary = { providerID: "claude-agent-sdk", modelID: "claude-haiku-5-5" };
const fallback = { providerID: "inception", modelID: "mercury-3", variant: "high" };
type Model = { providerID: string; id: string; variant?: string };

async function fixture(generate: (model: Model) => Promise<string>) {
  const calls: Model[] = [];
  const client = createLegacyClient({
    generate: {
      text: async ({ model }: { model: Model }) => {
        calls.push(model);
        return { text: await generate(model) };
      },
    },
  } as unknown as Context);
  const provider = await import(`../src/services/ai/opencode-provider.ts?fallback=${randomUUID()}`);
  provider.setV2Client(client);
  provider.setStructuredOutputFallback({ ...primary, fallback });
  const request = {
    client,
    ...primary,
    variant: "low",
    systemPrompt: "Return JSON",
    userPrompt: "A small test",
    schema: z.object({ summary: z.string() }),
  };
  return { provider, request, calls };
}

describe("structured-output model fallback", () => {
  it("keeps successful Haiku requests on the primary model", async () => {
    const f = await fixture(async () => '{"summary":"primary"}');
    expect(await f.provider.generateStructuredOutput(f.request)).toEqual({ summary: "primary" });
    expect(f.calls).toEqual([
      { providerID: primary.providerID, id: primary.modelID, variant: "low" },
    ]);
  });

  it("falls back once to Mercury High after a primary quota failure", async () => {
    const f = await fixture(async (model) => {
      if (model.providerID === primary.providerID) {
        throw new Error("Claude SDK: rate_limit: You've hit your 5-hour limit");
      }
      return '{"summary":"fallback"}';
    });
    expect(await f.provider.generateStructuredOutput(f.request)).toEqual({ summary: "fallback" });
    expect(f.calls).toEqual([
      { providerID: primary.providerID, id: primary.modelID, variant: "low" },
      { providerID: fallback.providerID, id: fallback.modelID, variant: "high" },
    ]);
  });

  it("allows the configured fallback through the disconnected-primary gate", async () => {
    const f = await fixture(async () => '{"summary":"ok"}');
    f.provider.setConnectedProviders([fallback.providerID]);
    expect(await f.provider.ensureProviderConnected(primary.providerID)).toBe(true);
    expect(f.provider.isProviderConnected(primary.providerID)).toBe(false);
    expect(await f.provider.ensureProviderConnected("unrelated")).toBe(false);
  });

  it("falls back when primary output fails schema validation", async () => {
    const f = await fixture(async (model) =>
      model.providerID === primary.providerID ? '{"summary":42}' : '{"summary":"valid"}'
    );
    expect(await f.provider.generateStructuredOutput(f.request)).toEqual({ summary: "valid" });
    expect(f.calls).toHaveLength(2);
  });

  it("propagates the fallback error without recursively retrying", async () => {
    const error = new Error("Mercury unavailable");
    const f = await fixture(async (model) => {
      if (model.providerID === primary.providerID) throw new Error("Haiku unavailable");
      throw error;
    });
    await expect(f.provider.generateStructuredOutput(f.request)).rejects.toBe(error);
    expect(f.calls).toHaveLength(2);
  });

  it("does not redirect other models or inherit the primary variant", async () => {
    const error = new Error("unavailable");
    const f = await fixture(async () => {
      throw error;
    });
    await expect(
      f.provider.generateStructuredOutput({ ...f.request, modelID: "another-model" })
    ).rejects.toBe(error);
    expect(f.calls).toHaveLength(1);

    f.provider.setStructuredOutputFallback({
      ...primary,
      fallback: { providerID: fallback.providerID, modelID: fallback.modelID },
    });
    await expect(f.provider.generateStructuredOutput(f.request)).rejects.toBe(error);
    expect(f.calls[2]).toEqual({ providerID: fallback.providerID, id: fallback.modelID });
  });

  it("disables the fallback when configuration is cleared", async () => {
    const error = new Error("unavailable");
    const f = await fixture(async () => {
      throw error;
    });
    f.provider.setStructuredOutputFallback();
    await expect(f.provider.generateStructuredOutput(f.request)).rejects.toBe(error);
    expect(f.calls).toHaveLength(1);
  });
});
