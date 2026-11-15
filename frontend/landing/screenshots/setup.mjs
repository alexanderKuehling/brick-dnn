/**
 * The data behind the landing page's screenshots, made through the REST API and the MCP server
 * of a local Brick DNN (see README.md): two accounts, a messy table of field trials with a
 * preparation, trainings on it and on built-in data, a search, a shared network with a comment
 * and some calls of an "AI assistant". Everything is looked up by name first, so running it
 * again reuses what is there and only adds what is missing.
 */

export const PASSWORD = 'landing-screenshots-2026';
export const USERS = {
  main: { username: 'ada', email: 'ada@example.com' },
  other: { username: 'grace', email: 'grace@example.com' },
};

export const NAMES = {
  source: 'Field trials 2021–2024',
  prep: 'Field trials: high yield',
  cnn: 'Small CNN (MNIST)',
  mlp: 'MLP: high yield',
  boosting: 'Gradient boosting search: high yield',
  unet: 'U-Net (Shapes)',
  hybrid: 'Hybrid: MLP features + gradient boosting',
  shared: 'Residual block (group work)',
  token: 'Claude (MCP)',
};

// ---------------------------------------------------------------------------------------------
// HTTP

export class Api {
  constructor(base, user) {
    this.base = base.replace(/\/+$/, '');
    this.user = user;
    this.tokens = null;
  }

  async login() {
    let res = await fetch(`${this.base}/api/auth/token/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: this.user.username, password: PASSWORD }),
    });
    if (res.status === 401 || res.status === 400) {
      res = await fetch(`${this.base}/api/auth/register/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...this.user, password: PASSWORD, accept_terms: true }),
      });
    }
    if (!res.ok) {
      throw new Error(`Login of ${this.user.username}: ${res.status} ${await res.text()}`);
    }
    const data = await res.json();
    this.tokens = { access: data.access, refresh: data.refresh };
    return this.tokens;
  }

  async request(method, path, body, { form = false, token = null, retry = true } = {}) {
    if (!this.tokens && !token) {
      await this.login();
    }
    const headers = { Authorization: `Bearer ${token ?? this.tokens.access}` };
    if (body !== undefined && !form) {
      headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(`${this.base}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : form ? body : JSON.stringify(body),
    });
    if (res.status === 401 && retry && !token) {
      await this.login();
      return this.request(method, path, body, { form, retry: false });
    }
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`${method} ${path}: ${res.status} ${text.slice(0, 500)}`);
    }
    return text ? JSON.parse(text) : null;
  }

  get(path) {
    return this.request('GET', path);
  }

  post(path, body, options) {
    return this.request('POST', path, body ?? {}, options);
  }
}

const list = (data) => (Array.isArray(data) ? data : (data?.results ?? []));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------------------------
// A messy table: field trials of a crop, whether a plot gave a high yield

/** Seeded random numbers (mulberry32), so the table is the same every time. */
function random(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function fieldTrialsCsv(rows = 900) {
  const rand = random(7);
  const normal = (mean, sd) =>
    mean + sd * Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const pick = (values) => values[Math.floor(rand() * values.length)];
  const lines = [
    'plot_id,sowing_date,site,soil,variety,irrigated,rainfall_mm,mean_temp_c,nitrogen_kg_ha,seed_rate,high_yield',
  ];
  const soilEffect = { loam: 0.6, silt: 0.3, clay: -0.2, sand: -0.7 };
  const varietyEffect = { 'A-114': 0.2, 'B-207': 0.5, 'C-031': -0.4 };
  for (let i = 1; i <= rows; i++) {
    const year = 2021 + Math.floor(rand() * 4);
    const day = new Date(Date.UTC(year, 2, 10) + Math.floor(rand() * 60) * 86400000);
    const site = pick(['North', 'South', 'East', 'West']);
    const soil = pick(['loam', 'loam', 'clay', 'sand', 'silt']);
    const variety = pick(['A-114', 'B-207', 'C-031']);
    const irrigated = rand() < 0.35 ? 'yes' : 'no';
    const rain = normal(site === 'South' ? 380 : 440, 85);
    const temp = normal(site === 'North' ? 14.2 : 15.8, 1.9);
    let nitrogen = 60 + rand() * 140;
    const seedRate = normal(320, 35);
    const z =
      -0.9 +
      0.006 * (rain - 420) +
      (irrigated === 'yes' ? 0.9 : 0) +
      0.35 * (temp - 15) -
      0.08 * (temp - 15) ** 2 +
      0.012 * (nitrogen - 130) +
      soilEffect[soil] +
      varietyEffect[variety] +
      0.004 * (seedRate - 320) +
      normal(0, 0.7);
    const high = rand() < 1 / (1 + Math.exp(-2.2 * z)) ? 'yes' : 'no';
    if (rand() < 0.015) {
      nitrogen *= 10; // a typing error in the field book
    }
    const missing = (p, value) => (rand() < p ? '' : value);
    lines.push(
      [
        `P${String(i).padStart(4, '0')}`,
        day.toISOString().slice(0, 10),
        site,
        missing(0.03, soil),
        variety,
        irrigated,
        missing(0.07, rain.toFixed(1)),
        temp.toFixed(1),
        nitrogen.toFixed(0),
        missing(0.02, seedRate.toFixed(0)),
        high,
      ].join(',')
    );
  }
  return `${lines.join('\n')}\n`;
}

const PREP_STEPS = [
  { id: 's1', type: 'drop_columns', enabled: true, params: { columns: ['plot_id'] } },
  {
    id: 's2',
    type: 'datetime_features',
    enabled: true,
    params: { columns: ['sowing_date'], parts: ['year', 'dayofyear'], drop_original: true },
  },
  {
    id: 's3',
    type: 'impute',
    enabled: true,
    params: { columns: ['rainfall_mm', 'seed_rate'], strategy: 'median' },
  },
  {
    id: 's4',
    type: 'impute',
    enabled: true,
    params: { columns: ['soil'], strategy: 'most_frequent' },
  },
  {
    id: 's5',
    type: 'outliers',
    enabled: true,
    params: { columns: ['nitrogen_kg_ha'], method: 'iqr', iqr_k: 1.5, action: 'clip' },
  },
  {
    id: 's6',
    type: 'encode',
    enabled: true,
    params: { columns: ['site', 'soil', 'variety'], method: 'onehot' },
  },
];

const PREP_SETTINGS = {
  target: 'high_yield',
  task: 'auto',
  split: { val: 0.15, test: 0.15, seed: 42, stratify: true },
};

// ---------------------------------------------------------------------------------------------
// Networks

function mlp(features) {
  const node = (id, type, params, x, y) => ({ id, type, params, position: { x, y } });
  return {
    version: 1,
    nodes: [
      node('input', 'Input', { shape: [features] }, 0, 0),
      node('fc1', 'Linear', { out_features: 64 }, 260, 0),
      node('act1', 'Activation', { function: 'relu' }, 520, 0),
      node('drop1', 'Dropout', { p: 0.2 }, 780, 0),
      node('fc2', 'Linear', { out_features: 32 }, 0, 160),
      node('act2', 'Activation', { function: 'relu' }, 260, 160),
      node('out', 'Linear', { out_features: 2 }, 520, 160),
    ],
    edges: [
      ['input', 'fc1'],
      ['fc1', 'act1'],
      ['act1', 'drop1'],
      ['drop1', 'fc2'],
      ['fc2', 'act2'],
      ['act2', 'out'],
    ].map(([source, target]) => ({ source, target })),
  };
}

/** The graph with the fixes the training check proposes for this dataset applied. */
async function fitted(api, graph, dataset) {
  const check = await api.post('/api/training/check/', { graph, dataset });
  const nodes = graph.nodes.map((node) => {
    const fix = (check.fixes ?? []).find((f) => f.node === node.id);
    return fix ? { ...node, params: { ...node.params, ...fix.params } } : node;
  });
  return { ...graph, nodes };
}

// ---------------------------------------------------------------------------------------------

/** Starts a run unless one of that name exists, and waits until it ends (an account trains
 * one at a time). */
async function ensureRun(api, name, body, log) {
  const runs = list(await api.get('/api/training/runs/'));
  let run = runs.find((r) => r.name === name && !['failed', 'stopped'].includes(r.status));
  if (!run) {
    log(`  starting “${name}”`);
    run = await api.post('/api/training/runs/', { name, ...body });
  }
  return (await waitForRuns(api, [run.id], log))[0];
}

export async function waitForRuns(api, ids, log = console.log) {
  for (;;) {
    const runs = await Promise.all(ids.map((id) => api.get(`/api/training/runs/${id}/`)));
    const open = runs.filter((run) =>
      ['queued', 'starting', 'preparing', 'running'].includes(run.status)
    );
    if (!open.length) {
      for (const run of runs) {
        log(`  ${run.name}: ${run.status}`);
      }
      return runs;
    }
    log(`  waiting for ${open.map((run) => `${run.name} (${run.status})`).join(', ')}`);
    await sleep(10000);
  }
}

async function mcp(api, token, calls) {
  let id = 1;
  const send = (method, params) =>
    api.request('POST', '/api/mcp/', { jsonrpc: '2.0', id: id++, method, params }, { token });
  await send('initialize', {
    protocolVersion: '2025-06-18',
    capabilities: {},
    clientInfo: { name: 'claude-ai', version: '1.0' },
  });
  for (const [name, args] of calls) {
    await send('tools/call', { name, arguments: args });
  }
}

/** Creates (or finds) everything the screenshots show. Returns the ids they need. */
export async function setup(base, log = console.log) {
  const api = new Api(base, USERS.main);
  const other = new Api(base, USERS.other);
  await api.login();
  await other.login();
  const examples = await api.get('/api/examples/');
  const example = (id) => structuredClone(examples.find((e) => e.id === id).graph);

  // Data: the table, its preparation and the training dataset made from it.
  log('Data');
  let source = list(await api.get('/api/data/sources/')).find((s) => s.name === NAMES.source);
  if (!source) {
    const form = new FormData();
    form.append('file', new Blob([fieldTrialsCsv()], { type: 'text/csv' }), 'field_trials.csv');
    form.append('name', NAMES.source);
    source = await api.request('POST', '/api/data/sources/', form, { form: true });
  }
  let prep = list(await api.get('/api/data/preps/')).find((p) => p.name === NAMES.prep);
  if (!prep) {
    prep = await api.post('/api/data/preps/', {
      source: source.id,
      name: NAMES.prep,
      steps: PREP_STEPS,
      settings: PREP_SETTINGS,
    });
  }
  prep = await api.get(`/api/data/preps/${prep.id}/`);
  let datasetId = prep.datasets?.[0]?.id;
  if (!datasetId) {
    datasetId = (await api.post(`/api/data/preps/${prep.id}/export/`, { name: NAMES.prep })).id;
  }
  const tabular = { kind: 'upload', id: datasetId };

  // Trainings.
  log('Trainings');
  const cnn = await ensureRun(
    api,
    NAMES.cnn,
    {
      graph: example('cnn_mnist'),
      dataset: { kind: 'builtin', id: 'mnist' },
      config: { epochs: 6, batch_size: 64, max_samples: 12000, seed: 1, patience: 3 },
    },
    log
  );
  const mlpGraph = await fitted(api, mlp(10), tabular);
  const mlpRun = await ensureRun(
    api,
    NAMES.mlp,
    {
      graph: mlpGraph,
      dataset: tabular,
      config: { epochs: 40, batch_size: 32, learning_rate: 0.002, seed: 3, patience: 8 },
    },
    log
  );
  const boosting = await ensureRun(
    api,
    NAMES.boosting,
    {
      graph: await fitted(api, example('gradient_boosting_classifier'), tabular),
      dataset: tabular,
      config: { seed: 5 },
      tuning: {
        method: 'random',
        trials: 16,
        metric: 'pr_auc',
        cv_folds: 5,
        space: {
          learning_rate: { min: 0.02, max: 0.3 },
          min_samples_leaf: { min: 5, max: 60, scale: 'linear' },
          max_depth: [null, 3, 6],
          class_weight: ['none', 'balanced'],
        },
      },
    },
    log
  );
  const unet = await ensureRun(
    api,
    NAMES.unet,
    {
      graph: example('unet'),
      dataset: { kind: 'builtin', id: 'shapes' },
      config: { epochs: 14, batch_size: 8, learning_rate: 0.002, seed: 2 },
    },
    log
  );
  const runs = [cnn, mlpRun, boosting, unet];

  // A hybrid network for the builder (not trained).
  const hybrid = await fitted(api, example('hybrid_mlp_gradient_boosting'), tabular);

  // Working together: a network shared with a second account, with a comment on a layer.
  log('Sharing');
  let shared = list(await api.get('/api/projects/?scope=mine')).find(
    (p) => p.name === NAMES.shared
  );
  if (!shared) {
    shared = await api.post('/api/projects/', {
      name: NAMES.shared,
      graph: example('resnet_block'),
      description: 'Exercise 3: why does the skip connection need the same shape?',
    });
    await api.post(`/api/projects/${shared.id}/members/`, {
      user: USERS.other.username,
      role: 'editor',
    });
    const conv = shared.graph.nodes.find((n) => n.type === 'Conv2d');
    await other.post(`/api/projects/${shared.id}/comments/`, {
      text: 'Could we use padding "same" here, so that Add gets two tensors of the same size?',
      node_id: conv.id,
    });
  }

  // An AI assistant: an API token and a few real MCP calls with it.
  log('MCP');
  const tokens = list(await api.get('/api/auth/api-tokens/'));
  if (!tokens.some((t) => t.name === NAMES.token)) {
    const { token } = await api.post('/api/auth/api-tokens/', { name: NAMES.token });
    const graph = example('mlp_mnist');
    await mcp(api, token, [
      ['list_examples', {}],
      ['analyze_network', { graph }],
      ['list_data_preps', {}],
      ['get_data_prep', { id: prep.id }],
      ['save_project', { name: 'Hybrid network for the field trials', graph: hybrid }],
      ['list_training_runs', {}],
      ['get_training_run', { id: mlpRun.id }],
      ['generate_pytorch_code', { graph }],
    ]);
  }

  const byName = Object.fromEntries(runs.map((run) => [run.name, run]));
  return {
    users: { main: api, other },
    prep: prep.id,
    dataset: datasetId,
    mlpGraph,
    shared: await api.get(`/api/projects/${shared.id}/`),
    hybrid,
    runs: {
      cnn: byName[NAMES.cnn],
      mlp: byName[NAMES.mlp],
      boosting: byName[NAMES.boosting],
      unet: byName[NAMES.unet],
    },
  };
}
