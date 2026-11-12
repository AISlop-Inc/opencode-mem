import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type GenerationRecord = {
  requestID: string;
  startedAt: string;
  durationMs: number;
  model: { providerID: string; id: string; variant?: string } | null;
  status: "success" | "error";
  inputBytes: number;
  outputBytes: number | null;
};

/** Store measurements only: never prompts, generated memories or error bodies. */
export function recordGeneration(path: string | undefined, record: GenerationRecord): void {
  if (!path) return;
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(
      path,
      JSON.stringify({
        ...record,
        // The host's stateless Generate API currently returns text only.
        tokens: null,
        cost: null,
        usageStatus: "not_exposed_by_host",
      }) + "\n",
      { mode: 0o600 }
    );
  } catch {
    console.warn("opencode-mem: could not append generation ledger");
  }
}
