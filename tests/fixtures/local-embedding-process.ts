import { writeFileSync } from "node:fs";
import type { EmbeddingReply, EmbeddingRequest } from "../../src/services/embedding-process.js";

const send = (reply: EmbeddingReply) => process.send?.(reply);

process.on("message", (request: EmbeddingRequest) => {
  if (request.type === "warmup") {
    send({ type: "ready" });
    return;
  }
  if (request.input === "crash") process.exit(2);
  if (request.input.startsWith("block:")) {
    writeFileSync(request.input.slice("block:".length), "");
    // A blocked native inference call likewise cannot read any more IPC messages.
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0);
  }
  send({ type: "vector", vector: new Float32Array([process.pid, request.input.length]) });
});
