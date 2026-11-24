import { fork } from "node:child_process";
import type { ProgressInfo } from "@huggingface/transformers";
import type { LocalOnnxDtype } from "./embedding.js";

export type EmbeddingRequest =
  | { type: "warmup"; model: string; cacheDir: string; dtype?: LocalOnnxDtype }
  | { type: "embed"; input: string; pooling: "mean" | "cls" | "last_token" };

export type EmbeddingReply =
  | { type: "progress"; progress: ProgressInfo }
  | { type: "ready" }
  | { type: "vector"; vector: Float32Array }
  | { type: "error"; message: string };

type Result = Exclude<EmbeddingReply, { type: "progress" | "error" }>;
type Job = {
  request: EmbeddingRequest;
  progress?: (progress: ProgressInfo) => void;
  resolve: (value: Result) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const MAX_PENDING = 32;

export function startEmbeddingProcess(entrypoint: URL, onExit: () => void) {
  const child = fork(entrypoint, [], {
    // OpenCode's standalone executable can also run a plain Bun script.
    env: process.versions.bun ? { ...process.env, BUN_BE_BUN: "1" } : process.env,
    execArgv: [],
    serialization: "advanced",
    stdio: ["ignore", "ignore", "ignore", "ipc"],
  });
  child.unref();
  child.channel?.unref?.();

  const queue: Job[] = [];
  let active: Job | undefined;
  let failure: Error | undefined;
  const exited = new Promise<void>((resolve) => {
    child.once("close", () => {
      fail(new Error("Local embedding process exited unexpectedly"));
      onExit();
      resolve();
    });
  });

  function fail(error: Error) {
    if (failure) return;
    failure = error;
    if (active) {
      clearTimeout(active.timer);
      active.reject(error);
      active = undefined;
    }
    for (const job of queue.splice(0)) {
      clearTimeout(job.timer);
      job.reject(error);
    }
  }

  function stop(error: Error) {
    fail(error);
    // Keep explicit shutdown alive until the child and its IPC channel close.
    child.ref();
    child.channel?.ref?.();
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    return exited;
  }

  function next() {
    if (active || failure) return;
    active = queue.shift();
    if (!active) return;
    child.send(active.request, (error) => {
      if (error) void stop(error);
    });
  }

  child.on("error", (error) => void stop(error));
  child.on("disconnect", () => void stop(new Error("Local embedding process disconnected")));
  child.on("message", (reply: EmbeddingReply) => {
    if (!active) return;
    if (reply.type === "progress") {
      try {
        active.progress?.(reply.progress);
      } catch (error) {
        void stop(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    clearTimeout(active.timer);
    if (reply.type === "error") active.reject(new Error(reply.message));
    else active.resolve(reply);
    active = undefined;
    next();
  });

  return {
    run(
      request: EmbeddingRequest,
      timeoutMs: number,
      progress?: (progress: ProgressInfo) => void
    ): Promise<Result> {
      if (failure) return Promise.reject(failure);
      if (queue.length + Number(!!active) >= MAX_PENDING) {
        return Promise.reject(new Error("Local embedding queue is full"));
      }
      return new Promise((resolve, reject) => {
        const job: Job = {
          request,
          progress,
          resolve,
          reject,
          timer: setTimeout(() => {
            const error = new Error(
              `Local embedding ${request.type} timed out after ${Math.ceil(timeoutMs)}ms`
            );
            if (active === job) {
              void stop(error);
              return;
            }
            const index = queue.indexOf(job);
            if (index !== -1) queue.splice(index, 1);
            reject(error);
          }, timeoutMs),
        };
        queue.push(job);
        next();
      });
    },
    close: () => stop(new Error("Local embedding process was closed")),
  };
}
