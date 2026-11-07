import assert from "node:assert/strict";

const PACKAGE_NAME = "@aislop-inc/opencode-mem";

const pluginModule = await import(PACKAGE_NAME);
const tagsModule = await import(`${PACKAGE_NAME}/tags`);

assert.equal(typeof pluginModule.default, "object", "default export must be a plugin object");
assert.equal(pluginModule.default.id, PACKAGE_NAME, "plugin id must match package name");
assert.equal(typeof pluginModule.default.server, "function", "plugin server must be callable");

assert.equal(typeof tagsModule.getTags, "function", "getTags export must be callable");
assert.equal(
  typeof tagsModule.getProjectTagInfo,
  "function",
  "getProjectTagInfo export must be callable"
);
assert.equal(
  typeof tagsModule.getUserTagInfo,
  "function",
  "getUserTagInfo export must be callable"
);

console.log("opencode-mem package smoke test passed");
