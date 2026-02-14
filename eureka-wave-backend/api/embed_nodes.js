function toVector1D(x) {
  if (!Array.isArray(x) || x.length === 0) return null;

  if (typeof x[0] === "number") return x;

  if (Array.isArray(x[0]) && typeof x[0][0] === "number") {
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

  if (Array.isArray(x[0]) && Array.isArray(x[0][0])) {
    return toVector1D(x[0]);
  }

  return null;
}

export default async function handler(req, res) {
  try {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === "OPTIONS") return res.status(200).end();
    if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

    const { nodes } = req.body || {};
    if (!Array.isArray(nodes) || !nodes.length) {
      return res.status(400).json({ error: "Missing nodes" });
    }

    const HF_API_KEY = process.env.HF_API_KEY;
    if (!HF_API_KEY) {
      return res.status(500).json({ error: "Missing HF_API_KEY (HuggingFace token)" });
    }

    const MODEL = "sentence-transformers/all-MiniLM-L6-v2";
    const HF_URL = `https://router.huggingface.co/hf-inference/models/${MODEL}/pipeline/feature-extraction`;

    const inputs = nodes.map(n => `${n.name} | ${n.tag || ""} | ${(n.tags || []).join(" ")}`);

    // ⚠️ Para batch: HF suele devolver [vec, vec, ...]
    const r = await fetch(HF_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ inputs })
    });

    if (!r.ok) {
      const t = await r.text();
      return res.status(502).json({ error: "HF embeddings upstream error", detail: t });
    }

    const raw = await r.json();

    if (!Array.isArray(raw) || raw.length !== inputs.length) {
      return res.status(500).json({ error: "Unexpected HF response shape", raw });
    }

    const out = nodes.map((n, i) => {
      const vec = toVector1D(raw[i]);
      return { ...n, vec };
    });

    const dims = out[0]?.vec?.length || null;
    if (!dims) {
      return res.status(500).json({ error: "Could not parse vectors from HF", sample: raw[0] });
    }

    return res.status(200).json({ nodes: out, dims, provider: "huggingface-router", model: MODEL });
  } catch (e) {
    return res.status(500).json({ error: "Server error", detail: String(e) });
  }
}
