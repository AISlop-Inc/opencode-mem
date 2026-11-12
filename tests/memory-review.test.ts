import { describe, expect, it } from "bun:test";
import {
  MemoryReviewScheduler,
  buildReviewTranscript,
  dueReviewBucket,
  resolveReviewModel,
  sanitizeReviewCandidates,
} from "../src/services/memory-review.js";
import { shouldRefreshRecallOnPrompt } from "../src/services/memory-recall.js";

describe("frozen snapshot", () => {
  it("never refreshes recall on prompt when snapshot is frozen", () => {
    expect(shouldRefreshRecallOnPrompt("relevant", "first", "frozen")).toBe(false);
    expect(shouldRefreshRecallOnPrompt("both", "always", "frozen")).toBe(false);
  });

  it("keeps legacy behaviour when snapshot is live", () => {
    expect(shouldRefreshRecallOnPrompt("relevant", "first", "live")).toBe(true);
    expect(shouldRefreshRecallOnPrompt("recent", "first", "live")).toBe(false);
  });
});

describe("memory review scheduling", () => {
  it("is due only at new multiples of the interval", () => {
    expect(dueReviewBucket(9, 0, 10)).toBeNull();
    expect(dueReviewBucket(10, 0, 10)).toBe(1);
    expect(dueReviewBucket(10, 1, 10)).toBeNull();
    expect(dueReviewBucket(23, 1, 10)).toBe(2);
  });

  it("runs once per bucket and retries after a failure", async () => {
    const s = new MemoryReviewScheduler();
    for (let i = 1; i <= 10; i++) s.noteTurn("s", `m${i}`);
    let calls = 0;
    await expect(
      s.maybeRun("s", 10, async () => {
        calls++;
        throw new Error("boom");
      })
    ).rejects.toThrow("boom");
    expect(await s.maybeRun("s", 10, async () => void calls++)).toBe(true);
    expect(await s.maybeRun("s", 10, async () => void calls++)).toBe(false);
    expect(calls).toBe(2);
  });

  it("counts a message once and restores after restart", () => {
    const s = new MemoryReviewScheduler();
    s.noteTurn("s", "m1");
    expect(s.noteTurn("s", "m1")).toBe(1);
    const r = new MemoryReviewScheduler();
    r.restore("s", 10, 10);
    expect(dueReviewBucket(10, 1, 10)).toBeNull();
    expect(r.noteTurn("s", "m11")).toBe(11);
  });
});

describe("review model resolution", () => {
  it("prefers memoryReview settings and falls back per field", () => {
    expect(
      resolveReviewModel(
        { opencodeProvider: "anthropic", opencodeModel: "haiku" },
        { opencodeProvider: "openai", opencodeModel: "gpt", opencodeVariant: "high" }
      )
    ).toEqual({ providerID: "anthropic", modelID: "haiku", variant: "high" });
  });

  it("falls back entirely to the top-level model", () => {
    expect(
      resolveReviewModel({}, { opencodeProvider: "openai", opencodeModel: "gpt-4o-mini" })
    ).toEqual({ providerID: "openai", modelID: "gpt-4o-mini" });
  });

  it("resolves inherit from the session model or returns null", () => {
    const review = { opencodeModel: "inherit", opencodeProvider: "x" };
    expect(resolveReviewModel(review, {}, { providerID: "p", modelID: "m" })).toEqual({
      providerID: "p",
      modelID: "m",
    });
    expect(resolveReviewModel(review, {}, null)).toBeNull();
    expect(resolveReviewModel({}, {})).toBeNull();
  });
});

describe("review transcript and candidates", () => {
  it("keeps only authored text from user and assistant messages", () => {
    const text = buildReviewTranscript([
      { info: { role: "user" }, parts: [{ type: "text", text: "hello" }] },
      {
        info: { role: "assistant" },
        parts: [
          { type: "text", text: "hi" },
          { type: "text", text: "hidden", synthetic: true },
          { type: "tool", id: "t" },
        ],
      },
      { info: { role: "assistant", summary: true }, parts: [{ type: "text", text: "sum" }] },
    ]);
    expect(text).toContain("### User\nhello");
    expect(text).toContain("### Assistant\nhi");
    expect(text).not.toContain("hidden");
    expect(text).not.toContain("sum");
  });

  it("dedupes, bounds, and normalizes candidates", () => {
    const out = sanitizeReviewCandidates(
      [
        { content: "Uses bun", type: "user-preference", tags: ["Bun", " "] },
        { content: "uses bun", type: "x", tags: [] },
        { content: "  ", type: "x", tags: [] },
        { content: "Other", type: "weird", tags: ["a", "b", "c", "d", "e"] },
      ],
      2
    );
    expect(out.map((c) => c.content)).toEqual(["Uses bun", "Other"]);
    expect(out[0]!.tags).toEqual(["bun"]);
    expect(out[1]!.type).toBe("project-fact");
    expect(out[1]!.tags.length).toBe(4);
  });
});
