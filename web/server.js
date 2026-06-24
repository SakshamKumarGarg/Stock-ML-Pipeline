import express from "express";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const app = express();
const port = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));

async function listSymbols() {
  const dataDir = path.join(rootDir, "processed_data");
  const files = await readdir(dataDir);
  const symbols = files
    .map((file) => file.match(/^([A-Za-z0-9.-]+)_processed_.*\.csv$/)?.[1])
    .filter(Boolean);

  return [...new Set(symbols)].sort();
}

app.get("/api/symbols", async (_req, res) => {
  try {
    res.json({ symbols: await listSymbols() });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/dashboard/:symbol", async (req, res) => {
  try {
    const symbol = req.params.symbol.toUpperCase();
    if (!/^[A-Z0-9.-]+$/.test(symbol)) {
      res.status(400).json({ error: "Invalid stock symbol" });
      return;
    }

    const cachePath = path.join(__dirname, "cache", `${symbol}.json`);
    res.json(JSON.parse(await readFile(cachePath, "utf8")));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Stock ML dashboard running at http://localhost:${port}`);
});
