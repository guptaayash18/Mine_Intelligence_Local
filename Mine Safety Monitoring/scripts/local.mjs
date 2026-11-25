// One command for the whole local stack: Ollama (the local LLM) + the app.
// Usage: npm run local          (first run downloads the model, about 1 GB)
//        npm run local -- --no-open   (do not open the browser)
//        npm run local -- --no-pull   (never download the model)
//
// Steps: check Node → find Ollama → start it if it is not running → download
// the model if missing → install dependencies and build if needed → start the
// app on http://127.0.0.1:8080 → open the browser. Nothing leaves this computer.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.chdir(root);
if (existsSync(".env")) process.loadEnvFile(".env");

const args = new Set(process.argv.slice(2));
const isWin = process.platform === "win32";
const model = process.env.OLLAMA_MODEL || "qwen2.5:1.5b";
const baseUrl = (
  process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434"
).replace(/\/+$/, "");
const host = process.env.HOST || "127.0.0.1";
const port = process.env.PORT || "8080";
const appUrl = `http://${host.includes(":") ? `[${host}]` : host}:${port}`;

const say = (m) => console.log(m);
const ok = (m) => say(`  [OK] ${m}`);
const step = (m) => say(`  [..] ${m}`);
const warn = (m) => say(`  [!]  ${m}`);
function die(m) {
  say(`  [X]  ${m}`);
  process.exit(1);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

say("\nMine Intelligence Locally: starting the local stack\n");

// 1. Node
const [major] = process.versions.node.split(".").map(Number);
if (major < 22)
  die(`Node.js ${process.versions.node} is too old. Install Node.js 22 LTS.`);
ok(`Node.js ${process.versions.node}`);

// 2. Ollama binary
function findOllama() {
  const candidates = ["ollama"];
  if (isWin && process.env.LOCALAPPDATA)
    candidates.push(
      path.join(process.env.LOCALAPPDATA, "Programs", "Ollama", "ollama.exe"),
    );
  if (process.platform === "darwin")
    candidates.push("/Applications/Ollama.app/Contents/Resources/ollama");
  for (const c of candidates) {
    const r = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (r.status === 0) return c;
  }
  return null;
}
const ollama = findOllama();

async function ollamaUp() {
  try {
    const r = await fetch(`${baseUrl}/api/version`, {
      signal: AbortSignal.timeout(2000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

let ollamaProc = null;
let aiReady = false;
if (!ollama && !(await ollamaUp())) {
  warn("Ollama is not installed. The dashboard and forecasts still work, but");
  say("       the AI report needs it: https://ollama.com/download");
} else {
  // 3. Start Ollama when it is not already running
  if (await ollamaUp()) {
    ok("Ollama is running");
  } else {
    step("Starting Ollama");
    ollamaProc = spawn(ollama, ["serve"], {
      stdio: "ignore",
      detached: false,
      env: {
        OLLAMA_NUM_PARALLEL: "1",
        OLLAMA_MAX_LOADED_MODELS: "1",
        OLLAMA_FLASH_ATTENTION: "1",
        OLLAMA_KV_CACHE_TYPE: "q8_0",
        ...process.env,
      },
    });
    ollamaProc.on("error", () => {});
    for (let i = 0; i < 30 && !(await ollamaUp()); i++) await sleep(1000);
    if (await ollamaUp()) ok("Ollama started");
    else
      warn("Ollama did not start in 30 s. The AI report will be unavailable.");
  }

  // 4. Model
  if (await ollamaUp()) {
    const hasModel = async () => {
      try {
        const r = await fetch(`${baseUrl}/api/tags`, {
          signal: AbortSignal.timeout(4000),
        });
        const body = await r.json();
        return (body.models ?? []).some(
          (m) =>
            (m.name ?? m.model ?? "") === model ||
            (m.name ?? "").startsWith(`${model}:`),
        );
      } catch {
        return false;
      }
    };
    if (await hasModel()) {
      ok(`AI model ${model}`);
      aiReady = true;
    } else if (args.has("--no-pull")) {
      warn(`AI model ${model} is not downloaded (skipped: --no-pull).`);
    } else {
      step(`Downloading the AI model ${model} (about 1 GB, one time)`);
      const r = spawnSync(ollama ?? "ollama", ["pull", model], {
        stdio: "inherit",
      });
      if (r.status === 0 && (await hasModel())) {
        ok(`AI model ${model}`);
        aiReady = true;
      } else
        warn("The model download did not finish. Run `npm run local` again.");
    }
  }
}

// 5. Dependencies and build
const npm = isWin ? "npm.cmd" : "npm";
const run = (cmd, a) => spawnSync(cmd, a, { stdio: "inherit", shell: isWin });
if (!existsSync("node_modules")) {
  step("Installing app dependencies (one time, needs internet)");
  if (run(npm, ["ci", "--no-audit", "--no-fund"]).status !== 0)
    die("Installing dependencies failed. Check the internet connection.");
}

function newestMtime(dir) {
  let newest = 0;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    newest = Math.max(
      newest,
      e.isDirectory() ? newestMtime(p) : statSync(p).mtimeMs,
    );
  }
  return newest;
}
const built = ".output/server/index.mjs";
const stale =
  !existsSync(built) ||
  Math.max(newestMtime("src"), statSync("package.json").mtimeMs) >
    statSync(built).mtimeMs;
if (stale) {
  step("Building the app (about 30 s)");
  if (run(npm, ["run", "build"]).status !== 0)
    die("The build failed. See the messages above.");
} else ok("App build is up to date");

// 6. Run the app
say(
  `\n  =============================================================\n   Mine Intelligence Locally is running: ${appUrl}\n   AI report: ${aiReady ? `ready (${model})` : "not available (see above)"}\n   Press Ctrl+C to stop.\n  =============================================================\n`,
);
const app = spawn(process.execPath, ["scripts/start.mjs"], {
  stdio: "inherit",
});

function openBrowser() {
  const [cmd, a] = isWin
    ? ["cmd", ["/c", "start", "", appUrl]]
    : process.platform === "darwin"
      ? ["open", [appUrl]]
      : ["xdg-open", [appUrl]];
  try {
    spawn(cmd, a, { stdio: "ignore", detached: true }).on("error", () => {});
  } catch {
    /* the URL is printed above */
  }
}
if (!args.has("--no-open")) {
  for (let i = 0; i < 20; i++) {
    try {
      const r = await fetch(appUrl, { signal: AbortSignal.timeout(1000) });
      if (r.status < 500) break;
    } catch {
      /* not up yet */
    }
    await sleep(500);
  }
  openBrowser();
}

function shutdown() {
  app.kill();
  // Only stop an Ollama that this script started.
  if (ollamaProc) ollamaProc.kill();
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
app.on("exit", (code) => {
  if (ollamaProc) ollamaProc.kill();
  process.exit(code ?? 0);
});
