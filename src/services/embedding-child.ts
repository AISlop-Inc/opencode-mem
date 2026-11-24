import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { loadEmbeddingBackend } from "./embedding-backend.js";
import { formatOnnxruntimeInitError } from "./onnxruntime-resolve.js";
import type { EmbeddingReply, EmbeddingRequest } from "./embedding-process.js";

let model: FeatureExtractionPipeline | undefined;
const send = (reply: EmbeddingReply) => process.send?.(reply);

process.on("disconnect", () => process.exit(0));
process.on("message", async (request: EmbeddingRequest) => {
  try {
    if (request.type === "warmup") {
      const backend = loadEmbeddingBackend(request.cacheDir);
      model = await backend.pipeline("feature-extraction", request.model, {
        dtype: request.dtype,
        session_options: { intraOpNumThreads: 2, interOpNumThreads: 1 },
        progress_callback: (progress) => send({ type: "progress", progress }),
      });
      send({ type: "ready" });
      return;
    }

    if (!model) throw new Error("Local embedding model is not initialized");
    const output = await model(request.input, { pooling: request.pooling, normalize: true });
    const data = output.data;
    const vector = data instanceof Float32Array ? data : Float32Array.from(data, Number);
    send({ type: "vector", vector });
  } catch (error) {
    let message = error instanceof Error ? error.message : String(error);
    if (request.type === "warmup") {
      // If the package itself is missing, resolving its binding path also throws.
      try {
        message = formatOnnxruntimeInitError(error).message;
      } catch {
        // Keep the original module resolution error.
      }
    }
    send({ type: "error", message });
  }
});
