function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const av = a[i] || 0;
    const bv = b[i] || 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

function toVector1D(x) {
  if (!Array.isArray(x) || x.length === 0) return null;
  if (typeof x[0] === "number") return x;

  if (Array.isArray(x[0])) {
    const tokens = x;
    const dims = tokens[0].length;
    const out = new Array(dims).fill(0);
    let count = 0;
    for (const t of tokens) {
      if (!Array.isArray(t) || t.length !== dims) continue;
      for (let i = 0; i < dims; i++) out[i] += t[i] || 0;
      count++;
    }
    if (!count) return null;
    for (let i = 0; i < dims; i++) out[i] /= count;
    return out;
  }
  return null;
}

function normalizeIntentScores(scores) {
  const keys = Object.keys(scores);
  let sum = 0;
  for (const k of keys) sum += scores[k];
  if (sum <= 0) {
    const uni = 1 / keys.length;
    const out = {};
    for (const k of keys) out[k] = uni;
    return out;
  }
  const out = {};
  for (const k of keys) out[k] = scores[k] / sum;
  return out;
}

function inferIntent(query) {
  const q = query.toLowerCase();
  const scores = {
    informational: 0.2,
    navigational: 0.1,
    commercial: 0.2,
    transactional: 0.2,
    local: 0.1
  };
  if (/\b(what|how|why|guide|tutorial|meaning|definition|examples)\b/.test(q)) scores.informational += 0.8;
  if (/\b(best|top|vs|compare|review)\b/.test(q)) scores.commercial += 0.6;
  if (/\b(buy|price|coupon|deal|order|subscribe|quote|book)\b/.test(q)) scores.transactional += 0.8;
  if (/\b(near me|nearby|hours|open now|directions|map)\b/.test(q)) scores.local += 0.9;
  if (/\b(login|site:|homepage|official|contact)\b/.test(q)) scores.navigational += 0.8;
  return normalizeIntentScores(scores);
}

function extractEntities(query) {
  const ents = [];
  const q = query.trim();

  if (/eureka\s+amor/i.test(q)) ents.push({ name: "Eureka Amor", type: "PERSON", confidence: 0.85 });

  const placeMatch = q.match(/\b(argentina|buenos aires|cathedral city|palm springs)\b/i);
  if (placeMatch) ents.push({ name: placeMatch[0], type: "PLACE", confidence: 0.75 });

  const concepts = ["seo", "sxo", "quantum", "qubit", "encryption", "entity", "knowledge graph"];
  for (const c of concepts) {
    const re = new RegExp(`\\b${c.replace(" ", "\\s+")}\\b`, "i");
    if (re.test(q)) ents.push({ name: c.toUpperCase(), type: "CONCEPT", confidence: 0.7 });
  }

  const seen = new Set();
  return ents.filter(e => {
    const k = e.name.toLowerCase();
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

export default async function handler(req, res) {
  try {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === "OPTIONS") return res.status(200).end();
    if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

    const { query, nodes } = req.body || {};
    if (!query) return res.status(400).json({ error: "Missing query" });

    const HF_API_KEY = process.env.HF_API_KEY;
    if (!HF_API_KEY) return res.status(500).json({ error: "Missing HF_API_KEY (HuggingFace token)" });

    const MODEL = "sentence-transformers/all-MiniLM-L6-v2";
    const HF_URL = `https://api-inference.huggingface.co/pipeline/feature-extraction/${MODEL}`;

    const embResp = await fetch(HF_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${HF_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        inputs: query,
        options: { wait_for_model: true }
      })
    });

    if (!embResp.ok) {
      const t = await embResp.text();
      return res.status(502).json({ error: "HF embeddings error", detail: t });
    }

    const raw = await embResp.json();
    const qVec = toVector1D(raw);
    if (!qVec) return res.status(500).json({ error: "Could not parse HF embedding response", raw });

    const intents = inferIntent(query);
    const entities = extractEntities(query);

    let cluster = "general";
    if (intents.local > 0.35) cluster = "local-seo";
    else if (intents.transactional > 0.35) cluster = "conversion";
    else if (intents.commercial > 0.35) cluster = "comparison";
    else if (intents.informational > 0.35) cluster = "education";

    const next_queries = [`${query} examples`, `${query} best practices`, `${query} checklist`];

    let target = null;
    if (Array.isArray(nodes) && nodes.length && Array.isArray(nodes[0]?.vec)) {
      let best = -Infinity;
      for (const n of nodes) {
        const s = cosine(qVec, n.vec);
        if (s > best) {
          best = s;
          target = { id: n.id, name: n.name, tag: n.tag, score: s };
        }
      }
    }

    return res.status(200).json({
      intents,
      entities,
      cluster,
      next_queries,
      explanation: "HF feature-extraction embeddings + deterministic intent heuristics.",
      target,
      embedding_meta: { provider: "huggingface", model: "all-MiniLM-L6-v2", dims: qVec.length }
    });
  } catch (e) {
    return res.status(500).json({ error: "Server error", detail: String(e) });
  }
}
