import { prepareOnnxruntimeForTransformers } from "./onnxruntime-resolve.js";
import { createRuntimeRequire } from "./runtime-require.js";
import {
  getTransformersPackageSpecifier,
  resolveTransformersCjsEntry,
} from "./transformers-resolve.js";

type Transformers = typeof import("@huggingface/transformers");

export function loadEmbeddingBackend(cacheDir: string): Pick<Transformers, "pipeline" | "env"> {
  const requireFromHere = createRuntimeRequire(import.meta);
  // Resolve before installing the ONNX pin shim. OpenCode's compiled Bun host
  // can lose the referrer for non-ONNX package resolution after that point.
  const entry = resolveTransformersCjsEntry({
    requireFn: requireFromHere,
    specifier: getTransformersPackageSpecifier(),
    startPaths: [import.meta.url, import.meta.path, import.meta.dir, process.cwd()],
  });

  prepareOnnxruntimeForTransformers();
  const backend = requireFromHere(entry) as Transformers;
  backend.env.allowLocalModels = true;
  backend.env.allowRemoteModels = true;
  backend.env.cacheDir = cacheDir;
  const wasm = backend.env.backends.onnx.wasm;
  if (wasm) wasm.numThreads = 1;
  return backend;
}
