import { expect, it } from "bun:test";
import { existsSync, mkdtempSync, rmSync, watch } from "node:fs";
import { get } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startEmbeddingProcess } from "../src/services/embedding-process.js";

const fixture = new URL("./fixtures/local-embedding-process.ts", import.meta.url);
const embed = (input: string) => ({ type: "embed", input, pooling: "mean" }) as const;

it("reuses one child for concurrent requests and returns typed vectors", async () => {
  const child = startEmbeddingProcess(fixture, () => {});
  try {
    const replies = await Promise.all([
      child.run(embed("first"), 3000),
      child.run(embed("second"), 3000),
    ]);
    const vectors = replies.map((reply) => {
      if (reply.type !== "vector") throw new Error("Expected a vector");
      expect(reply.vector).toBeInstanceOf(Float32Array);
      return reply.vector;
    });
    expect(vectors[0]?.[0]).toBe(vectors[1]?.[0]);
    expect(vectors[0]?.[0]).not.toBe(process.pid);
    expect(vectors.map((vector) => vector[1])).toEqual([5, 6]);
  } finally {
    await child.close();
  }
});

it("keeps the host responsive during synchronous child work and kills the hung child", async () => {
  const child = startEmbeddingProcess(fixture, () => {});
  const server = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: () => new Response("ok") });
  const dir = mkdtempSync(join(tmpdir(), "embedding-process-"));
  const marker = join(dir, "started");
  const started = Promise.withResolvers<void>();
  const watcher = watch(dir, () => {
    if (existsSync(marker)) started.resolve();
  });
  try {
    const blocked = child.run(embed(`block:${marker}`), 1500).catch((error: Error) => error);
    expect(await Promise.race([started.promise, blocked])).toBeUndefined();
    // Test the host while the child is still blocked, not after it was killed.
    const probe = new Promise<number | undefined>((resolve, reject) => {
      get(server.url, (response) => {
        response.resume();
        response.on("end", () => resolve(response.statusCode));
      }).on("error", reject);
    });
    expect(await Promise.race([probe, blocked])).toBe(200);
    expect(await blocked).toBeInstanceOf(Error);
    expect(await blocked).toHaveProperty("message", expect.stringContaining("timed out"));
  } finally {
    watcher.close();
    server.stop(true);
    await child.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

it("rejects both active and queued work when the child exits", async () => {
  const child = startEmbeddingProcess(fixture, () => {});
  try {
    const first = child.run(embed("crash"), 3000);
    const second = child.run(embed("queued"), 3000);
    for (const reply of await Promise.allSettled([first, second])) {
      expect(reply.status).toBe("rejected");
      if (reply.status === "rejected") expect(reply.reason.message).toMatch(/exited|disconnected/);
    }
  } finally {
    await child.close();
  }
});
