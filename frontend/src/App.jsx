import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

const API = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? 'https://gan-backend.onrender.com' : 'http://127.0.0.1:8000')
).replace(/\/$/, '');

// ── tiny utilities ──
function debounce(fn, wait) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), wait); };
}

function Section({ title, subtitle, children, id }) {
  return (
    <div className="card" id={id}>
      <div className="card-header">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </div>
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function ImageCell({ src }) {
  return (
    <div className="image-cell">
      {src ? <img src={src} alt="" /> : <div style={{ width: '100%', height: '100%', background: 'rgba(255,255,255,0.02)' }} />}
    </div>
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      {payload.map((p, i) => (
        <p key={i}>
          <span className="tooltip-label">{p.name}: </span>
          <span className="tooltip-val">{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </p>
      ))}
    </div>
  );
}

// ─────────────────────────── THEORY ───────────────────────────
function TheorySection() {
  const classes = [
    [0, 'T-shirt / Top'], [1, 'Trouser'], [2, 'Pullover'], [3, 'Dress'], [4, 'Coat'],
    [5, 'Sandal'], [6, 'Shirt'], [7, 'Sneaker'], [8, 'Bag'], [9, 'Ankle boot'],
  ];
  return (
    <>
      <Section
        title="Introduction to GANs"
        subtitle="Goodfellow et al. (2014) — Generative Adversarial Networks"
      >
        <div className="theory-grid">
          <div className="theory-item full-width">
            <h4>Generative modeling</h4>
            <p>
              Generative modeling discovers the patterns in training data so the model can sample new
              plausible examples from the same distribution. Unlike classification, which asks
              "what is this?", generative models ask "what does the data look like?"
            </p>
          </div>

          <div className="theory-item">
            <h4>Two-player minimax game</h4>
            <p>
              A GAN is a pair of networks trained adversarially. The <strong style={{ color: 'var(--text-primary)' }}>Generator</strong> produces
              fake samples from noise; the <strong style={{ color: 'var(--text-primary)' }}>Discriminator</strong> classifies
              samples as real or fake. They are trained with opposing objectives — G minimizes what D maximizes.
            </p>
          </div>

          <div className="theory-item">
            <h4>Counterfeiter vs detective</h4>
            <p>
              Intuition: G is a counterfeiter printing fake money and D is police trying to detect it.
              Both improve over time — the counterfeiter gets better at faking, the detective gets sharper
              at spotting fakes. At equilibrium, fakes are indistinguishable from reals.
            </p>
          </div>
        </div>
      </Section>

      <Section
        title="Generator Network"
        subtitle="z → G(z): maps noise to an image that fools the discriminator"
      >
        <div className="theory-grid">
          <div className="theory-item">
            <h4>Input</h4>
            <p>Latent vector <span className="type-badge">z ~ N(0, I)</span> of dimension <code style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-primary)' }}>latent_dim</code>. Typically 64 or 100 dims for small images.</p>
          </div>
          <div className="theory-item">
            <h4>Architecture</h4>
            <p>Dense → BatchNorm → LeakyReLU → Dense → ... → Dense → Tanh. Final layer outputs 784 values reshaped to (28, 28, 1). Tanh maps outputs to [-1, 1] to match normalized inputs.</p>
          </div>
          <div className="theory-item">
            <h4>Objective</h4>
            <p>Maximize D(G(z)) — fool the discriminator into believing fakes are real. In practice we minimize the non-saturating loss <code style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-primary)' }}>-log D(G(z))</code> for stronger gradients early in training.</p>
          </div>
          <div className="theory-item">
            <h4>Why BatchNorm</h4>
            <p>Stabilizes training and helps the generator avoid mode collapse. It normalizes activations so D can't trivially detect fakes by their scale alone.</p>
          </div>
        </div>
      </Section>

      <Section
        title="Discriminator Network"
        subtitle="x → D(x) ∈ [0, 1]: probability that x is a real image"
      >
        <div className="theory-grid">
          <div className="theory-item">
            <h4>Input</h4>
            <p>An image <span className="type-badge">x ∈ ℝ^784</span> — either a real Fashion MNIST sample or a fake G(z). Flattened from (28, 28, 1).</p>
          </div>
          <div className="theory-item">
            <h4>Architecture</h4>
            <p>Dense → LeakyReLU → Dropout → ... → Dense → Sigmoid. Dropout prevents D from becoming too strong too fast — which would starve G of learning signal.</p>
          </div>
          <div className="theory-item">
            <h4>Objective</h4>
            <p>Maximize log D(x_real) + log(1 − D(G(z))). Assign high probability to real data and low probability to fakes.</p>
          </div>
          <div className="theory-item">
            <h4>Label smoothing</h4>
            <p>Replacing real target 1.0 with 0.9 prevents D from becoming overconfident and producing vanishing gradients for G. A standard stabilization trick.</p>
          </div>
        </div>
      </Section>

      <Section
        title="Minimax Objective"
        subtitle="The value function V(D, G) — what the two networks compete over"
      >
        <div className="formula">
{`min_G  max_D  V(D, G) = E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 − D(G(z)))]

Discriminator loss  L_D = − [ log D(x_real) + log(1 − D(G(z))) ]
Generator loss      L_G = − log D(G(z))                               (non-saturating)
`}
        </div>
        <p style={{ color: 'var(--text-secondary)', marginTop: 16, lineHeight: 1.7 }}>
          At the global optimum, D(x) = 1/2 everywhere — the discriminator cannot distinguish real
          from fake, meaning the generator has perfectly matched the data distribution. In practice
          we never converge there, but the adversarial dynamic drives G toward it.
        </p>
      </Section>

      <Section
        title="Training Loop"
        subtitle="Alternating gradient updates on D then G each batch"
      >
        <div className="algorithm">
          <p><strong>for</strong> each epoch:</p>
          <p className="indent1"><strong>for</strong> each batch of real images x:</p>
          <p className="indent2"><strong>// Train Discriminator</strong></p>
          <p className="indent2">z ~ N(0, I)</p>
          <p className="indent2">x_fake = G(z)</p>
          <p className="indent2">L_D = BCE(D(x_real), 1) + BCE(D(x_fake), 0)</p>
          <p className="indent2">update D weights via ∇L_D</p>
          <p className="indent2"> </p>
          <p className="indent2"><strong>// Train Generator</strong></p>
          <p className="indent2">z ~ N(0, I)</p>
          <p className="indent2">x_fake = G(z)</p>
          <p className="indent2">L_G = BCE(D(x_fake), 1)   <span style={{ color: 'var(--text-muted)' }}>// pretend fake is real</span></p>
          <p className="indent2">update G weights via ∇L_G</p>
        </div>
      </Section>

      <Section title="GAN vs VAE vs Autoencoder" subtitle="How the three generative approaches compare">
        <div className="theory-item full-width" style={{ padding: 0, background: 'transparent', border: 'none' }}>
          <table className="theory-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Training Objective</th>
                <th>Sampling</th>
                <th>Output Quality</th>
                <th>Mode Coverage</th>
                <th>Stability</th>
                <th>Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono">GAN</td>
                <td>Adversarial minimax</td>
                <td>G(z), z ~ N(0, I)</td>
                <td>Sharp, high-fidelity</td>
                <td>Prone to collapse</td>
                <td>Difficult</td>
                <td>Realistic image generation</td>
              </tr>
              <tr>
                <td className="mono">VAE</td>
                <td>ELBO (recon + KL)</td>
                <td>Decode z ~ N(0, I)</td>
                <td>Blurry but coherent</td>
                <td>Good coverage</td>
                <td>Easy</td>
                <td>Smooth latent interpolation</td>
              </tr>
              <tr>
                <td className="mono">AE</td>
                <td>Reconstruction loss</td>
                <td>Not generative</td>
                <td>Faithful to inputs</td>
                <td>N/A</td>
                <td>Very easy</td>
                <td>Compression, denoising</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Fashion MNIST Classes" subtitle="Ten clothing categories, 28×28 grayscale, 60k train / 10k test">
        <div className="theory-item full-width" style={{ padding: 0, background: 'transparent', border: 'none' }}>
          <table className="theory-table">
            <thead>
              <tr><th style={{ width: 120 }}>Class Index</th><th>Label</th></tr>
            </thead>
            <tbody>
              {classes.map(([idx, label]) => (
                <tr key={idx}>
                  <td className="mono">{idx}</td>
                  <td>{label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

// ─────────────────────────── ARCH BUILDER ───────────────────────────
const G_HIDDEN_PRESETS = [[256], [128, 256], [128, 256, 512]];
const D_HIDDEN_PRESETS = [[256], [512, 256], [512, 256, 128]];

function PillSelector({ options, value, onChange, format = v => v }) {
  const same = (a, b) => Array.isArray(a) && Array.isArray(b)
    ? a.length === b.length && a.every((v, i) => v === b[i])
    : a === b;
  return (
    <div className="arch-row-pills">
      {options.map((opt, i) => (
        <button
          key={i}
          className={`pill-tab ${same(opt, value) ? 'active' : ''}`}
          onClick={() => onChange(opt)}
        >
          {format(opt)}
        </button>
      ))}
    </div>
  );
}

function ArchitectureDiagram({ latentDim, gHidden, dHidden }) {
  // horizontal SVG — two rows: top for Generator, bottom for Discriminator.
  const boxW = 74, boxH = 52, gap = 64, padX = 30, rowGap = 80;
  const gLayers = [
    { label: `z`, sub: `${latentDim}`, kind: 'in' },
    ...gHidden.map(h => ({ label: `${h}`, sub: 'LeakyReLU', kind: 'hidden' })),
    { label: '784', sub: 'Tanh', kind: 'out' },
    { label: '28×28', sub: 'reshape', kind: 'out' },
  ];
  const dLayers = [
    { label: '784', sub: 'input', kind: 'in' },
    ...dHidden.map(h => ({ label: `${h}`, sub: 'LeakyReLU', kind: 'hidden' })),
    { label: '1', sub: 'Sigmoid', kind: 'out' },
  ];
  const maxLen = Math.max(gLayers.length, dLayers.length);
  const width = padX * 2 + maxLen * boxW + (maxLen - 1) * gap;
  const height = rowGap * 2 + boxH + 40;

  const renderRow = (layers, y, title) => {
    const startX = padX;
    return (
      <g>
        <text x={startX} y={y - 14} fill="#6b7280" fontSize="11" fontFamily="'JetBrains Mono', monospace" style={{ letterSpacing: '0.08em' }}>
          {title}
        </text>
        {layers.map((L, i) => {
          const x = startX + i * (boxW + gap);
          return (
            <g key={i}>
              <rect x={x} y={y} width={boxW} height={boxH} rx={8}
                    fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.14)" />
              <text x={x + boxW / 2} y={y + 22} textAnchor="middle"
                    fill="#ffffff" fontSize="13" fontWeight={600} fontFamily="'JetBrains Mono', monospace">
                {L.label}
              </text>
              <text x={x + boxW / 2} y={y + 40} textAnchor="middle"
                    fill="#6b7280" fontSize="10" fontFamily="'JetBrains Mono', monospace">
                {L.sub}
              </text>
              {i < layers.length - 1 && (
                <path d={`M ${x + boxW} ${y + boxH / 2} L ${x + boxW + gap} ${y + boxH / 2}`}
                      stroke="rgba(255,255,255,0.25)" strokeWidth="1.4" markerEnd="url(#arrow)" fill="none" />
              )}
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <div className="arch-diagram">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.35)" />
          </marker>
        </defs>
        {renderRow(gLayers, 30, 'GENERATOR')}
        {renderRow(dLayers, 30 + rowGap, 'DISCRIMINATOR')}
      </svg>
    </div>
  );
}

function ArchitectureBuilder({ config, setConfig }) {
  const update = (k, v) => setConfig(c => ({ ...c, [k]: v }));
  const fmtArr = a => `[${a.join(', ')}]`;
  return (
    <>
      <div className="arch-panels">
        <div className="arch-panel">
          <h4>Generator</h4>
          <div className="arch-row">
            <span className="arch-row-label">Latent dim z</span>
            <PillSelector options={[16, 32, 64, 128]} value={config.latent_dim}
                          onChange={v => update('latent_dim', v)} />
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Hidden layers</span>
            <PillSelector options={G_HIDDEN_PRESETS} value={config.g_hidden}
                          onChange={v => update('g_hidden', v)} format={fmtArr} />
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Activation</span>
            <PillSelector options={['leaky', 'relu']} value={config.g_activation}
                          onChange={v => update('g_activation', v)}
                          format={v => v === 'leaky' ? 'LeakyReLU' : 'ReLU'} />
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Output activation</span>
            <span className="arch-row-val">Tanh</span>
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Output shape</span>
            <span className="arch-row-val">784 → (28, 28, 1)</span>
          </div>
        </div>

        <div className="arch-panel">
          <h4>Discriminator</h4>
          <div className="arch-row">
            <span className="arch-row-label">Input shape</span>
            <span className="arch-row-val">784</span>
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Hidden layers</span>
            <PillSelector options={D_HIDDEN_PRESETS} value={config.d_hidden}
                          onChange={v => update('d_hidden', v)} format={fmtArr} />
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Activation</span>
            <span className="arch-row-val">LeakyReLU</span>
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Dropout</span>
            <PillSelector options={[0.1, 0.2, 0.3]} value={config.dropout}
                          onChange={v => update('dropout', v)} format={v => v.toFixed(1)} />
          </div>
          <div className="arch-row">
            <span className="arch-row-label">Output</span>
            <span className="arch-row-val">Sigmoid</span>
          </div>
        </div>
      </div>

      <ArchitectureDiagram latentDim={config.latent_dim} gHidden={config.g_hidden} dHidden={config.d_hidden} />
    </>
  );
}

// ─────────────────────────── TRAIN ───────────────────────────
function HyperparamControls({ config, setConfig }) {
  const update = (k, v) => setConfig(c => ({ ...c, [k]: v }));
  return (
    <>
      <div className="controls-row">
        <div className="control-group">
          <label>Epochs <strong>{config.epochs}</strong></label>
          <input type="range" min={1} max={50} step={1}
                 value={config.epochs}
                 onChange={e => update('epochs', parseInt(e.target.value))} />
          <div className="range-labels"><span>1</span><span>50</span></div>
        </div>
        <div className="control-group">
          <label>Batch size <strong>{config.batch_size}</strong></label>
          <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
            {[16, 32, 64, 128].map(v => (
              <button key={v} className={`pill-tab ${config.batch_size === v ? 'active' : ''}`}
                      onClick={() => update('batch_size', v)}>{v}</button>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label>Learning rate <strong>{config.lr}</strong></label>
          <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
            {[0.0001, 0.0002, 0.0005, 0.001].map(v => (
              <button key={v} className={`pill-tab ${config.lr === v ? 'active' : ''}`}
                      onClick={() => update('lr', v)}>{v}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="controls-row">
        <div className="control-group">
          <label>Optimizer <strong>{config.optimizer}</strong></label>
          <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
            {['adam', 'rmsprop', 'sgd'].map(v => (
              <button key={v} className={`pill-tab ${config.optimizer === v ? 'active' : ''}`}
                      onClick={() => update('optimizer', v)}>{v}</button>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label>Label smoothing <strong>{config.label_smoothing.toFixed(2)}</strong></label>
          <input type="range" min={0} max={0.2} step={0.01}
                 value={config.label_smoothing}
                 onChange={e => update('label_smoothing', parseFloat(e.target.value))} />
          <div className="range-labels"><span>0.00</span><span>0.20</span></div>
        </div>
        <div className="control-group">
          <label>Sample interval <strong>{config.sample_interval}</strong></label>
          <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
            {[10, 25, 50, 100].map(v => (
              <button key={v} className={`pill-tab ${config.sample_interval === v ? 'active' : ''}`}
                      onClick={() => update('sample_interval', v)}>{v}</button>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

function TrainSection({ config, setConfig, status, setStatus, lossLog, setLossLog, training, setTraining }) {
  const pollRef = useRef(null);

  const startTraining = async () => {
    setLossLog([]);
    setTraining(true);
    try {
      await fetch(`${API}/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
    } catch (e) {
      console.error(e);
      setTraining(false);
    }
  };

  const stopTraining = async () => {
    try { await fetch(`${API}/train/stop`, { method: 'POST' }); } catch (e) { console.error(e); }
    setTraining(false);
  };

  const resetTraining = async () => {
    try { await fetch(`${API}/train/reset`, { method: 'POST' }); } catch (e) { console.error(e); }
    setTraining(false);
    setLossLog([]);
    setStatus(null);
  };

  // polling
  useEffect(() => {
    if (!training) return;
    const tick = async () => {
      try {
        const r = await fetch(`${API}/status`);
        const data = await r.json();
        setStatus(data);
        if (data.g_loss && data.d_loss) {
          const merged = data.g_loss.map(p => {
            const d = data.d_loss.find(x => x.step === p.step);
            return { step: p.step, g: p.val, d: d ? d.val : 0 };
          });
          setLossLog(merged);
        }
        if (data.done || data.error || !data.running) {
          setTraining(false);
        }
      } catch (e) {
        console.error(e);
      }
    };
    tick();
    pollRef.current = setInterval(tick, 1500);
    return () => clearInterval(pollRef.current);
  }, [training, setStatus, setLossLog, setTraining]);

  const progress = status?.total_epochs
    ? ((status.epoch - 1 + (status.batch / (status.total_batches || 1))) / status.total_epochs) * 100
    : 0;

  const statusText = !status ? 'Idle'
    : status.error ? `Error: ${status.error}`
    : status.running ? `Training... Epoch ${status.epoch}/${status.total_epochs} · Batch ${status.batch}/${status.total_batches}`
    : status.done ? `Done · ${status.total_epochs} epochs completed`
    : 'Idle';

  const statusClass = status?.running ? 'running' : status?.error ? 'error' : status?.done ? 'done' : '';

  const currentGLoss = lossLog.length ? lossLog[lossLog.length - 1].g.toFixed(4) : '—';
  const currentDLoss = lossLog.length ? lossLog[lossLog.length - 1].d.toFixed(4) : '—';
  const currentDx = status?.d_x?.length ? status.d_x[status.d_x.length - 1].val.toFixed(3) : '—';
  const currentDgz = status?.d_gz?.length ? status.d_gz[status.d_gz.length - 1].val.toFixed(3) : '—';
  const currentEpoch = status?.epoch || 0;

  const images = status?.images || [];

  return (
    <>
      <Section title="Architecture Builder" subtitle="Configure both networks — the diagram updates live">
        <ArchitectureBuilder config={config} setConfig={setConfig} />
      </Section>

      <Section title="Hyperparameters" subtitle="Training regime and stabilization options">
        <HyperparamControls config={config} setConfig={setConfig} />
      </Section>

      <Section title="Training" subtitle="Runs on the backend as a daemon thread — polled every 1.5s">
        <div className="train-controls">
          <button className="btn-primary" onClick={startTraining} disabled={training}>
            {training ? <span className="spinner"></span> : null}
            {training ? 'Training...' : 'Train'}
          </button>
          <button className="btn-sm" onClick={stopTraining} disabled={!training}>Stop</button>
          <button className="btn-sm" onClick={resetTraining} disabled={training}>Reset</button>
          <div className="train-status">
            <span className={`status-dot ${statusClass}`}></span>
            <span>{statusText}</span>
          </div>
        </div>
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
        </div>

        <div className="results-grid">
          <div className="result-card">
            <div className="label">Epoch</div>
            <div className="value">{currentEpoch}<span style={{ color: 'var(--text-muted)' }}>/{status?.total_epochs || config.epochs}</span></div>
          </div>
          <div className="result-card">
            <div className="label">G Loss</div>
            <div className="value">{currentGLoss}</div>
          </div>
          <div className="result-card">
            <div className="label">D Loss</div>
            <div className="value">{currentDLoss}</div>
          </div>
          <div className="result-card">
            <div className="label">D(x) / D(G(z))</div>
            <div className="value">{currentDx} / {currentDgz}</div>
          </div>
        </div>

        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={lossLog}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" minTickGap={30} tickMargin={10}
                     label={{ value: 'step', position: 'insideBottom', offset: -4, fill: '#6b7280', fontSize: 12 }} />
              <YAxis stroke="#64748b" tickMargin={10} width={50} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: '#a3a3a3' }} />
              <Line type="monotone" dataKey="g" name="G Loss" stroke="#ffffff" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="d" name="D Loss" stroke="#6b7280" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="image-epoch-label">
          Generated samples · Epoch {currentEpoch || '—'} · Batch {status?.batch || '—'}
        </div>
        <div className="image-grid">
          {Array.from({ length: 32 }).map((_, i) => (
            <ImageCell key={i} src={images[i]} />
          ))}
        </div>
      </Section>
    </>
  );
}

// ─────────────────────────── GENERATOR TAB ───────────────────────────
const PERFECT_SQUARES = [4, 9, 16, 25, 36, 49, 64];

function GeneratorSection({ status }) {
  const [nSamples, setNSamples] = useState(16);
  const [temperature, setTemperature] = useState(1.0);
  const [genImages, setGenImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const [interpSteps, setInterpSteps] = useState(8);
  const [interpImages, setInterpImages] = useState([]);
  const [interpLoading, setInterpLoading] = useState(false);

  const [inspect, setInspect] = useState(null);
  const [inspectLoading, setInspectLoading] = useState(false);

  const hasModel = status?.has_model;

  const generate = async () => {
    if (!hasModel) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: nSamples, temperature }),
      });
      const d = await r.json();
      if (d.images) setGenImages(d.images);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const interpolate = async () => {
    if (!hasModel) return;
    setInterpLoading(true);
    try {
      const r = await fetch(`${API}/interpolate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: interpSteps }),
      });
      const d = await r.json();
      if (d.images) setInterpImages(d.images);
    } catch (e) { console.error(e); }
    setInterpLoading(false);
  };

  const inspectRandom = async () => {
    if (!hasModel) return;
    setInspectLoading(true);
    try {
      const r = await fetch(`${API}/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const d = await r.json();
      if (!d.error) setInspect(d);
    } catch (e) { console.error(e); }
    setInspectLoading(false);
  };

  const cols = Math.sqrt(nSamples);

  if (!hasModel) {
    return (
      <Section title="Generator" subtitle="Train a model in the Train tab first">
        <div className="notice">
          <strong>No trained model available.</strong> Go to <em>Train</em>, configure your architecture,
          and click <em>Train</em>. Once training completes, return here to generate images,
          explore latent space interpolation, and inspect noise vectors.
        </div>
      </Section>
    );
  }

  return (
    <>
      <Section title="Batch Generator" subtitle="Sample images from the trained generator">
        <div className="gen-controls">
          <div className="control-group">
            <label>Samples <strong>{nSamples}</strong> ({cols}×{cols} grid)</label>
            <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
              {PERFECT_SQUARES.map(n => (
                <button key={n} className={`pill-tab ${nSamples === n ? 'active' : ''}`}
                        onClick={() => setNSamples(n)}>{n}</button>
              ))}
            </div>
          </div>
          <div className="control-group">
            <label>Temperature <strong>{temperature.toFixed(2)}</strong></label>
            <input type="range" min={0.1} max={2.0} step={0.05}
                   value={temperature}
                   onChange={e => setTemperature(parseFloat(e.target.value))} />
            <div className="range-labels"><span>0.10</span><span>2.00</span></div>
          </div>
          <div>
            <button className="btn-primary" onClick={generate} disabled={loading}>
              {loading ? <span className="spinner"></span> : null}
              {loading ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </div>

        <div className="gen-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {genImages.length
            ? genImages.map((src, i) => <div key={i} className="gen-cell"><img src={src} alt="" /></div>)
            : Array.from({ length: nSamples }).map((_, i) => (
                <div key={i} className="gen-cell" style={{ background: 'var(--bg-secondary)' }} />
              ))}
        </div>
      </Section>

      <Section title="Latent Space Interpolation" subtitle="Walk from noise vector A to B — see how output morphs smoothly">
        <div className="gen-controls">
          <div className="control-group">
            <label>Steps <strong>{interpSteps}</strong></label>
            <div className="arch-row-pills" style={{ justifyContent: 'flex-start' }}>
              {[5, 8, 10, 12].map(n => (
                <button key={n} className={`pill-tab ${interpSteps === n ? 'active' : ''}`}
                        onClick={() => setInterpSteps(n)}>{n}</button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn-primary" onClick={interpolate} disabled={interpLoading}>
              {interpLoading ? <span className="spinner"></span> : null}
              {interpLoading ? 'Interpolating...' : 'Interpolate'}
            </button>
          </div>
        </div>

        <div className="interp-strip">
          {interpImages.length === 0 && Array.from({ length: interpSteps }).map((_, i) => (
            <div key={i} className="interp-cell">
              <div style={{ width: 64, height: 64, background: 'var(--bg-secondary)', borderRadius: 4, border: '1px solid var(--border)' }} />
              <div className="interp-cell-label">{i === 0 ? 'A' : i === interpSteps - 1 ? 'B' : ''}</div>
            </div>
          ))}
          {interpImages.map((src, i) => (
            <div key={i} className={`interp-cell ${i === 0 || i === interpImages.length - 1 ? 'endpoint' : ''}`}>
              <img src={src} alt="" />
              <div className="interp-cell-label">
                {i === 0 ? 'A' : i === interpImages.length - 1 ? 'B' : `${(i / (interpImages.length - 1)).toFixed(2)}`}
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Noise Vector Inspector" subtitle="See the exact z used to produce an image, visualized as dimension bars">
        <div className="inspector-actions" style={{ marginBottom: 24 }}>
          <button className="btn-sm" onClick={inspectRandom} disabled={inspectLoading}>
            {inspectLoading ? <span className="spinner spinner-light"></span> : null}
            {inspect ? 'New random z' : 'Generate random z'}
          </button>
        </div>

        {inspect ? (
          <div className="inspector-layout">
            <div>
              <img className="inspector-image" src={inspect.image} alt="" />
              <div style={{ marginTop: 12, color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'JetBrains Mono, monospace' }}>
                latent_dim = {inspect.latent_dim}
              </div>
            </div>
            <div className="noise-bars">
              {inspect.z.slice(0, Math.min(inspect.z.length, 64)).map((v, i) => {
                const mag = Math.min(Math.abs(v) / 3, 1) * 50;
                return (
                  <div key={i} className="noise-bar-row">
                    <span className="noise-label">z{i}</span>
                    <div className="noise-track">
                      <div className="noise-center" />
                      {v >= 0
                        ? <div className="noise-fill noise-fill-pos" style={{ width: `${mag}%` }} />
                        : <div className="noise-fill noise-fill-neg" style={{ width: `${mag}%` }} />}
                    </div>
                    <span className="noise-val">{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="notice">Click <strong>Generate random z</strong> to sample a noise vector and visualize it.</div>
        )}
      </Section>
    </>
  );
}

// ─────────────────────────── ANALYSE TAB ───────────────────────────
function AnalyseSection({ status, lossLog }) {
  const [realImages, setRealImages] = useState([]);
  const [fakeImages, setFakeImages] = useState([]);
  const [metricView, setMetricView] = useState('loss');
  const [diversity, setDiversity] = useState(null);
  const [divLoading, setDivLoading] = useState(false);
  const [snapshots, setSnapshots] = useState([]);
  const [scrubEpoch, setScrubEpoch] = useState(1);

  const hasModel = status?.has_model;

  const fetchReal = useCallback(async () => {
    try {
      const r = await fetch(`${API}/real-samples`);
      const d = await r.json();
      if (d.images) setRealImages(d.images);
    } catch (e) { console.error(e); }
  }, []);

  const fetchFakes = useCallback(async () => {
    if (!hasModel) return;
    try {
      const r = await fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 8, temperature: 1.0 }),
      });
      const d = await r.json();
      if (d.images) setFakeImages(d.images);
    } catch (e) { console.error(e); }
  }, [hasModel]);

  const fetchDiversity = async () => {
    if (!hasModel) return;
    setDivLoading(true);
    try {
      const r = await fetch(`${API}/diversity`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const d = await r.json();
      setDiversity(d);
    } catch (e) { console.error(e); }
    setDivLoading(false);
  };

  const fetchSnapshots = useCallback(async () => {
    try {
      const r = await fetch(`${API}/snapshots`);
      const d = await r.json();
      if (d.snapshots) {
        setSnapshots(d.snapshots);
        if (d.snapshots.length) setScrubEpoch(d.snapshots.length);
      }
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { fetchReal(); }, [fetchReal]);
  useEffect(() => { if (hasModel) { fetchFakes(); fetchSnapshots(); } }, [hasModel, fetchFakes, fetchSnapshots]);

  const chartData = useMemo(() => {
    if (metricView === 'loss') return lossLog.map(p => ({ step: p.step, a: p.g, b: p.d }));
    if (metricView === 'dx')   return (status?.d_x || []).map(p => ({ step: p.step, a: p.val, b: null }));
    if (metricView === 'dgz')  return (status?.d_gz || []).map(p => ({ step: p.step, a: p.val, b: null }));
    return [];
  }, [metricView, lossLog, status]);

  const chartLabels = {
    loss: { a: 'G Loss', b: 'D Loss' },
    dx:   { a: 'D(x) — real score', b: null },
    dgz:  { a: 'D(G(z)) — fake score', b: null },
  }[metricView];

  const scrubSnap = snapshots.find(s => s.epoch === scrubEpoch);

  return (
    <>
      <Section title="Real vs Fake" subtitle="Eight real Fashion MNIST samples alongside eight from your generator">
        <div className="compare-split">
          <div>
            <div className="compare-col-header">Real</div>
            <div className="img-row">
              {realImages.length
                ? realImages.slice(0, 8).map((src, i) => <ImageCell key={i} src={src} />)
                : Array.from({ length: 8 }).map((_, i) => <ImageCell key={i} src={null} />)}
            </div>
          </div>
          <div>
            <div className="compare-col-header">Generated</div>
            <div className="img-row">
              {fakeImages.length
                ? fakeImages.slice(0, 8).map((src, i) => <ImageCell key={i} src={src} />)
                : Array.from({ length: 8 }).map((_, i) => <ImageCell key={i} src={null} />)}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn-sm" onClick={fetchFakes} disabled={!hasModel}>Refresh Fakes</button>
          <button className="btn-sm" onClick={fetchReal}>Refresh Reals</button>
        </div>
      </Section>

      <Section title="Training History" subtitle="Full record of losses and discriminator scores across the run">
        <div className="pill-tabs">
          {[
            { id: 'loss', label: 'Loss' },
            { id: 'dx', label: 'D(x) Score' },
            { id: 'dgz', label: 'D(G(z)) Score' },
          ].map(t => (
            <button key={t.id} className={`pill-tab ${metricView === t.id ? 'active' : ''}`}
                    onClick={() => setMetricView(t.id)}>{t.label}</button>
          ))}
        </div>

        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" minTickGap={30} tickMargin={10} />
              <YAxis stroke="#64748b" tickMargin={10} width={50} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: '#a3a3a3' }} />
              <Line type="monotone" dataKey="a" name={chartLabels.a} stroke="#ffffff" strokeWidth={2} dot={false} isAnimationActive={false} />
              {chartLabels.b && (
                <Line type="monotone" dataKey="b" name={chartLabels.b} stroke="#6b7280" strokeWidth={2} dot={false} isAnimationActive={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Mode Collapse Detector" subtitle="Average pairwise pixel distance across 16 generated samples — higher means more diverse">
        <div className="diversity-card">
          <h4>Diversity Analysis</h4>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            If the generator has collapsed to a single mode, all outputs look alike and the pairwise
            distance approaches zero.
          </p>

          {diversity && !diversity.error ? (
            <>
              <div className="diversity-meter">
                <div className="diversity-track">
                  <div className="diversity-fill" style={{ width: `${(diversity.normalized || 0) * 100}%` }} />
                </div>
                <div className="diversity-label">{(diversity.score).toFixed(3)}</div>
              </div>
              <div className="diversity-verdict">{diversity.verdict}</div>
              <div className="diversity-grid">
                {diversity.images.slice(0, 16).map((src, i) => <ImageCell key={i} src={src} />)}
              </div>
            </>
          ) : (
            <div className="notice" style={{ marginTop: 16 }}>
              {hasModel ? 'Click Compute Diversity to analyse.' : 'Train a model first.'}
            </div>
          )}

          <div style={{ marginTop: 20 }}>
            <button className="btn-sm" onClick={fetchDiversity} disabled={!hasModel || divLoading}>
              {divLoading ? <span className="spinner spinner-light"></span> : null}
              {diversity ? 'Recompute' : 'Compute Diversity'}
            </button>
          </div>
        </div>
      </Section>

      <Section title="Epoch Playback" subtitle="Scrub through samples saved at each epoch — watch quality improve over training">
        {snapshots.length === 0 ? (
          <div className="notice">No snapshots yet. Train a model to populate this view.</div>
        ) : (
          <>
            <div className="epoch-scrubber">
              <label>Epoch <strong>{scrubEpoch}</strong> / {snapshots.length}</label>
              <input type="range" min={1} max={snapshots.length} step={1}
                     value={scrubEpoch}
                     onChange={e => setScrubEpoch(parseInt(e.target.value))} />
              <div className="range-labels"><span>1</span><span>{snapshots.length}</span></div>
            </div>
            <div className="image-grid" style={{ gridTemplateColumns: 'repeat(8, 1fr)' }}>
              {(scrubSnap?.images || []).map((src, i) => <ImageCell key={i} src={src} />)}
              {Array.from({ length: Math.max(0, 8 - (scrubSnap?.images.length || 0)) }).map((_, i) => (
                <ImageCell key={`pad-${i}`} src={null} />
              ))}
            </div>
            <div style={{ marginTop: 12 }}>
              <button className="btn-sm" onClick={fetchSnapshots}>Refresh snapshots</button>
            </div>
          </>
        )}
      </Section>
    </>
  );
}

// ─────────────────────────── FOOTER ───────────────────────────
function Footer() {
  return (
    <footer className="app-footer">
      <span className="footer-copy">© Viraj Choudhary</span>
      <div className="footer-links">
        <a href="https://github.com/virajchoudhary" target="_blank" rel="noopener noreferrer">GitHub</a>
        <a href="https://www.linkedin.com/in/virajchoudhary" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <a href="https://x.com/virajchoudhary_" target="_blank" rel="noopener noreferrer">Twitter</a>
        <a href="mailto:virajc188@gmail.com">Email</a>
      </div>
    </footer>
  );
}

// ─────────────────────────── APP ───────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('theory');

  const [config, setConfig] = useState({
    latent_dim: 64,
    g_hidden: [128, 256],
    d_hidden: [512, 256],
    g_activation: 'leaky',
    dropout: 0.2,
    epochs: 10,
    batch_size: 32,
    lr: 0.0002,
    optimizer: 'adam',
    label_smoothing: 0.1,
    sample_interval: 50,
  });

  const [status, setStatus] = useState(null);
  const [lossLog, setLossLog] = useState([]);
  const [training, setTraining] = useState(false);

  // on mount — ping /health and /status once so GeneratorSection knows if a model already exists
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/status`);
        const d = await r.json();
        setStatus(d);
        if (d.g_loss && d.d_loss) {
          const merged = d.g_loss.map(p => {
            const dd = d.d_loss.find(x => x.step === p.step);
            return { step: p.step, g: p.val, d: dd ? dd.val : 0 };
          });
          setLossLog(merged);
        }
      } catch (e) { /* backend may not be up yet */ }
    })();
  }, []);

  const tabs = [
    { id: 'theory',    label: 'Theory' },
    { id: 'train',     label: 'Train' },
    { id: 'generator', label: 'Generator' },
    { id: 'analyse',   label: 'Analyse' },
  ];

  return (
    <div className="app-layout">
      <nav className="sidebar">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`sidebar-item ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="main-content">
        <header className="app-header">
          <h1>Generative Adversarial Network</h1>
          <p>Goodfellow-style GAN on Fashion MNIST — build the architecture, train live, explore latent space.</p>
        </header>

        <div className="page-content">
          {activeTab === 'theory' && <TheorySection />}
          {activeTab === 'train' && (
            <TrainSection
              config={config} setConfig={setConfig}
              status={status} setStatus={setStatus}
              lossLog={lossLog} setLossLog={setLossLog}
              training={training} setTraining={setTraining}
            />
          )}
          {activeTab === 'generator' && <GeneratorSection status={status} />}
          {activeTab === 'analyse' && <AnalyseSection status={status} lossLog={lossLog} />}
        </div>

        <Footer />
      </main>
    </div>
  );
}
