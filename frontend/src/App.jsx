import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import './App.css';
import { API_BASE_URL } from './config';

function debounce(func, wait) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

async function fetchWithTimeout(url, options = {}, timeout = 10000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        return res;
    } catch (err) {
        clearTimeout(id);
        throw err;
    }
}

async function fetchWithRetry(url, options = {}, retries = 3, onRetry) {
    try {
        const res = await fetchWithTimeout(url, options);
        if (!res.ok) throw new Error("API error");
        return res;
    } catch (e) {
        if (retries === 0) throw e;
        if (onRetry) onRetry(retries);
        await new Promise(r => setTimeout(r, 3000));
        return fetchWithRetry(url, options, retries - 1, onRetry);
    }
}

// ─── Static Section ───────────────────────────────────────────────────────────
function Section({ title, subtitle, children, id }) {
    return (
        <div className="card" id={id}>
            <div className="card-header" style={{ cursor: 'default' }}>
                <div>
                    <h2>{title}</h2>
                    {subtitle && <p className="card-subtitle">{subtitle}</p>}
                </div>
            </div>
            <div className="card-body">{children}</div>
        </div>
    );
}

// ─── Image Generator ────────────────────────────────────────────────────────
function ImageGenerator() {
    const [numImages, setNumImages] = useState(16);
    const [truncation, setTruncation] = useState(1.0);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const generate = async () => {
        setLoading(true);
        try {
            const res = await fetchWithRetry(`${API_BASE_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_images: numImages, truncation })
            });
            setResult(await res.json());
        } catch (e) {
            console.error(e);
            alert("Generator unavailable. The backend might be waking up or experiencing memory limits.");
        }
        setLoading(false);
    };

    return (
        <Section title="Image Generator" subtitle="Generate CIFAR-10 images from random latent vectors" id="generator">
            <div className="controls-row">
                <div className="control-group">
                    <label>Number of Images: <strong>{numImages}</strong></label>
                    <input type="range" min="1" max="64" step="1" value={numImages}
                        onChange={e => setNumImages(parseInt(e.target.value))} />
                    <div className="range-labels"><span>1</span><span>64</span></div>
                </div>
                <div className="control-group">
                    <label>Truncation (σ): <strong>{truncation.toFixed(1)}</strong></label>
                    <input type="range" min="0.1" max="2.0" step="0.1" value={truncation}
                        onChange={e => setTruncation(parseFloat(e.target.value))} />
                    <div className="range-labels"><span>0.1 (sharp)</span><span>2.0 (diverse)</span></div>
                </div>
                <button className="btn-primary" onClick={generate} disabled={loading}>
                    {loading ? <span className="spinner"></span> : null}
                    {loading ? 'Generating...' : 'Generate'}
                </button>
            </div>
            {result && (
                <div className="gen-results fade-in">
                    <div className="stats-bar">
                        <span className="stat">Images: <strong>{result.num_generated}</strong></span>
                        <span className="stat">Avg Critic Score: <strong className="accent">{result.avg_score}</strong></span>
                        <span className="stat">Truncation: <strong>{result.truncation}</strong></span>
                    </div>
                    <div className="image-card-grid">
                        {result.images.map((imgSrc, i) => (
                            <div key={i} className="image-card">
                                <img src={imgSrc} alt={`Sample ${i}`} />
                                <div className="card-score">
                                    <span className={`chip ${result.scores[i] > 0 ? 'positive' : 'negative'}`}>
                                        Score: {result.scores[i]}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </Section>
    );
}

// ─── Latent Space Explorer (WebSocket) ──────────────────────────────────────
function LatentExplorer() {
    const [genImage, setGenImage] = useState(null);
    const [z, setZ] = useState(new Array(100).fill(0));

    const abortRef = useRef(null);

    const debouncedGenerate = useRef(debounce(async (vector) => {
        if (abortRef.current) abortRef.current.abort();
        abortRef.current = new AbortController();

        try {
            const res = await fetch(`${API_BASE_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ z: vector, truncation: 1.0 }),
                signal: abortRef.current.signal
            });
            if (!res.ok) throw new Error("Backend error");
            const data = await res.json();
            if (data.images && data.images.length > 0) {
                setGenImage(data.images[0]);
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Fetch error:', error);
            }
        }
    }, 300)).current;

    const updateLocalZ = (idx, val) => {
        const newZ = [...z];
        for (let i = 0; i < 10; i++) newZ[idx * 10 + i] = parseFloat(val);
        setZ(newZ);
        return newZ;
    };

    const randomize = () => {
        const newZ = Array.from({ length: 100 }, () => (Math.random() * 2 - 1) * 2);
        setZ(newZ);
        debouncedGenerate(newZ);
    };

    const reset = () => {
        const newZ = new Array(100).fill(0);
        setZ(newZ);
        debouncedGenerate(newZ);
    };

    return (
        <Section title="Latent Space Explorer" subtitle="Manipulate latent dimensions in real-time" id="latent-explorer">
            <div className="explorer-layout">
                <div className="sliders-panel">
                    <div className="slider-actions">
                        <button className="btn-sm" onClick={randomize}>Randomize</button>
                        <button className="btn-sm btn-outline" onClick={reset}>Reset</button>
                    </div>
                    <div className="sliders">
                        {Array.from({ length: 10 }).map((_, i) => (
                            <div key={i} className="slider-row">
                                <span className="slider-label">z[{i * 10}‒{i * 10 + 9}]</span>
                                <input type="range" min="-3" max="3" step="0.1"
                                    value={z[i * 10]}
                                    onChange={(e) => {
                                        const currentVector = updateLocalZ(i, e.target.value);
                                        debouncedGenerate(currentVector);
                                    }} />
                                <span className="slider-val">{z[i * 10].toFixed(1)}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="preview-panel">
                    <div className="image-display">
                        {genImage ? <img src={genImage} alt="Generated" /> : <div className="placeholder">Move sliders to generate</div>}
                    </div>
                    <p className="preview-hint">32×32 CIFAR-10 output</p>
                </div>
            </div>
        </Section>
    );
}

// ─── Latent Interpolation ───────────────────────────────────────────────────
function Interpolation() {
    const [steps, setSteps] = useState(8);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const interpolate = async () => {
        setLoading(true);
        try {
            const res = await fetchWithRetry(`${API_BASE_URL}/interpolate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps })
            });
            setResult(await res.json());
        } catch (e) {
            console.error(e);
            alert("Interpolation failed. The backend might be starting up.");
        }
        setLoading(false);
    };

    return (
        <Section title="Latent Space Interpolation" subtitle="Spherical interpolation (SLERP) between two random z-vectors" id="interpolation">
            <div className="controls-row">
                <div className="control-group">
                    <label>Interpolation Steps: <strong>{steps}</strong></label>
                    <input type="range" min="3" max="20" step="1" value={steps}
                        onChange={e => setSteps(parseInt(e.target.value))} />
                    <div className="range-labels"><span>3</span><span>20</span></div>
                </div>
                <button className="btn-primary" onClick={interpolate} disabled={loading}>
                    {loading ? <span className="spinner"></span> : null}
                    {loading ? 'Interpolating...' : 'Interpolate'}
                </button>
            </div>
            {result && (
                <div className="interp-results fade-in">
                    <div className="interp-strip">
                        {result.images.map((imgUrl, i) => (
                            <img key={i} src={imgUrl} alt={`Frame ${i}`} />
                        ))}
                    </div>
                    <div className="interp-labels">
                        <span>z₁ (random)</span>
                        <span className="interp-arrow">→ SLERP →</span>
                        <span>z₂ (random)</span>
                    </div>
                </div>
            )}
        </Section>
    );
}

// ─── Architecture Viewer ────────────────────────────────────────────────────
function ArchitectureViewer() {
    const [info, setInfo] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE_URL}/model-info`).then(r => r.json()).then(setInfo).catch(() => {});
    }, []);

    const formatNum = n => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : n;

    if (!info) return <Section title="Architecture Viewer" subtitle="Loading model info..." id="architecture"><div className="loading-bar"></div></Section>;

    const maxLayers = Math.max(info.generator.layers.length, info.critic.layers.length);

    return (
        <Section title="Architecture Viewer" subtitle={`z_dim=${info.z_dim} | image_size=${info.image_size}×${info.image_size} | Dataset: ${info.training_info.dataset}`} id="architecture">
            <div className="arch-model" style={{ marginBottom: '32px' }}>
                <table className="arch-table">
                    <thead>
                        <tr>
                            <th colSpan="4" style={{ borderRight: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
                                Generator ({formatNum(info.generator.total_params)} params)
                            </th>
                            <th colSpan="4" style={{ textAlign: 'center' }}>
                                Critic ({formatNum(info.critic.total_params)} params)
                            </th>
                        </tr>
                        <tr>
                            <th style={{ width: '10%' }}>Layer</th>
                            <th style={{ width: '12%' }}>Type</th>
                            <th style={{ width: '10%' }}>Params</th>
                            <th style={{ width: '18%', borderRight: '1px solid rgba(255,255,255,0.08)' }}>Details</th>
                            
                            <th style={{ width: '10%', paddingLeft: '16px' }}>Layer</th>
                            <th style={{ width: '12%' }}>Type</th>
                            <th style={{ width: '10%' }}>Params</th>
                            <th style={{ width: '18%' }}>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: maxLayers }).map((_, i) => {
                            const g = info.generator.layers[i] || { name: '-', type: '-', params: 0, details: '-' };
                            const d = info.critic.layers[i] || { name: '-', type: '-', params: 0, details: '-' };
                            
                            const clamp = str => str.length > 30 ? str.slice(0, 30) + '...' : str;

                            return (
                                <tr key={i}>
                                    <td className="mono">{g.name}</td>
                                    <td>{g.type !== '-' ? <span className="type-badge">{g.type}</span> : '-'}</td>
                                    <td>{g.params > 0 ? formatNum(g.params) : '-'}</td>
                                    <td className="mono detail-cell" style={{borderRight: '1px solid rgba(255,255,255,0.08)', paddingRight: '16px'}}>{clamp(g.details)}</td>
                                    
                                    <td className="mono" style={{paddingLeft: '16px'}}>{d.name}</td>
                                    <td>{d.type !== '-' ? <span className="type-badge">{d.type}</span> : '-'}</td>
                                    <td>{d.params > 0 ? formatNum(d.params) : '-'}</td>
                                    <td className="mono detail-cell">{clamp(d.details)}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            <div className="training-config">
                <h4>Training Configuration</h4>
                <div className="config-chips">
                    <span className="config-chip">Optimizer: {info.training_info.optimizer}</span>
                    <span className="config-chip">LR: {info.training_info.lr}</span>
                    <span className="config-chip">λ_GP: {info.training_info.lambda_gp}</span>
                    <span className="config-chip">n_critic: {info.training_info.n_critic}</span>
                </div>
            </div>
        </Section>
    );
}

// ─── WGAN-GP Theory ─────────────────────────────────────────────────────────
function WGANTheory() {
    return (
        <Section title="Introduction to the Wasserstein GAN" subtitle="A beginner-friendly overview of Generative Adversarial Networks and the Wasserstein Distance." defaultOpen={true} id="theory">
            <div className="theory-grid">
                <div className="theory-item full-width">
                    <h4>What is a GAN?</h4>
                    <p>A <strong>Generative Adversarial Network (GAN)</strong> is a type of machine learning framework where two neural networks are pitted against each other in a continuous game. The <strong>Generator</strong> tries to create fake data (like images of cars or animals) that look as realistic as possible. Meanwhile, the <strong>Discriminator</strong> examines images and tries to guess whether they are real (from your actual dataset) or fake (created by the generator).</p>
                    <p>Through this competition, the generator becomes incredibly good at producing highly realistic, synthetic data. However, standard GANs are famously unstable and difficult to train. That's where the <strong>Wasserstein GAN (WGAN)</strong> comes in.</p>
                </div>
                <div className="theory-item">
                    <h4>Wasserstein Distance</h4>
                    <p>Unlike standard GANs that use JS divergence, WGAN uses the <strong>Earth Mover's Distance</strong> (Wasserstein-1 distance) to measure the distance between the real and generated distributions:</p>
                    <div className="formula">W(P_r, P_g) = inf<sub>γ∈Π(P_r,P_g)</sub> E<sub>(x,y)~γ</sub>[‖x − y‖]</div>
                    <p>This provides <strong>meaningful gradients</strong> even when the distributions don't overlap, solving the vanishing gradient problem of standard GANs.</p>
                </div>
                <div className="theory-item">
                    <h4>Critic vs Discriminator</h4>
                    <p>In standard GANs, the discriminator outputs a probability ∈ [0,1]. In WGAN, the <strong>critic</strong> outputs an unbounded real number — a "realness score". Higher scores = more realistic.</p>
                    <div className="formula">L_critic = E[D(x̃)] − E[D(x)] <br/>L_generator = −E[D(x̃)]</div>
                    <p>The critic tries to <strong>maximize</strong> the difference between real and fake scores; the generator tries to <strong>maximize</strong> its score.</p>
                </div>
                <div className="theory-item">
                    <h4>1-Lipschitz Constraint</h4>
                    <p>The Wasserstein distance requires the critic to be <strong>1-Lipschitz continuous</strong>: |f(x₁) − f(x₂)| ≤ ‖x₁ − x₂‖ for all x₁, x₂.</p>
                    <ul>
                        <li><strong>WGAN (original)</strong>: enforces via weight clipping → causes capacity issues</li>
                        <li><strong>WGAN-GP</strong>: enforces via gradient penalty → much better training</li>
                    </ul>
                </div>
                <div className="theory-item">
                    <h4>Gradient Penalty</h4>
                    <p>Instead of clipping weights, WGAN-GP adds a <strong>penalty term</strong> to the critic loss that pushes gradient norms toward 1:</p>
                    <div className="formula">GP = λ · E<sub>x̂</sub>[(‖∇<sub>x̂</sub>D(x̂)‖₂ − 1)²]</div>
                    <p>Where x̂ is sampled uniformly along lines between real and generated samples. Typically <strong>λ = 10</strong> and the critic is updated <strong>5 times</strong> per generator step.</p>
                </div>
                <div className="theory-item full-width">
                    <h4>WGAN Training Algorithm</h4>
                    <div className="algorithm">
                        <p><strong>for</strong> each training iteration:</p>
                        <p className="indent1"><strong>for</strong> t = 1, ..., n_critic:</p>
                        <p className="indent2">Sample real batch x ~ P_data, noise z ~ N(0, I)</p>
                        <p className="indent2">x̃ ← G(z), ε ~ U(0,1), x̂ ← ε·x + (1−ε)·x̃</p>
                        <p className="indent2">L_D ← D(x̃) − D(x) + λ(‖∇_{'{x̂}'}D(x̂)‖₂ − 1)²</p>
                        <p className="indent2">Update critic: θ_D ← θ_D − α·∇L_D</p>
                        <p className="indent1">Sample z ~ N(0, I)</p>
                        <p className="indent1">L_G ← −D(G(z))</p>
                        <p className="indent1">Update generator: θ_G ← θ_G − α·∇L_G</p>
                    </div>
                </div>
                <div className="theory-item full-width">
                    <h4>Key Advantages over Standard GANs</h4>
                    <table className="theory-table">
                        <thead><tr><th>Aspect</th><th>Standard GAN</th><th>Wasserstein GAN</th></tr></thead>
                        <tbody>
                            <tr><td>Loss</td><td>JS Divergence / BCE</td><td>Wasserstein Distance</td></tr>
                            <tr><td>Training Signal</td><td>Vanishes when D is too good</td><td>Always meaningful gradients</td></tr>
                            <tr><td>Mode Collapse</td><td>Common</td><td>Significantly reduced</td></tr>
                            <tr><td>D Output</td><td>Probability [0, 1]</td><td>Unbounded score (critic)</td></tr>
                            <tr><td>Constraint</td><td>None</td><td>1-Lipschitz via gradient penalty</td></tr>
                            <tr><td>Stability</td><td>Sensitive to hyperparams</td><td>Much more stable training</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </Section>
    );
}

// ─── Training Metrics ───────────────────────────────────────────────────────
function TrainingMetrics() {
    const [logs, setLogs] = useState([]);

    useEffect(() => {
        fetch(`${API_BASE_URL}/logs`)
            .then(r => r.json())
            .then(data => {
                const formatted = data.critic_loss.map((c, i) => ({
                    epoch: i + 1, critic: parseFloat(c.toFixed(4)), gen: parseFloat(data.gen_loss[i].toFixed(4))
                }));
                setLogs(formatted);
            }).catch(() => {});
    }, []);

    const finalCritic = logs.length > 0 ? logs[logs.length - 1].critic : 0;
    const finalGen = logs.length > 0 ? logs[logs.length - 1].gen : 0;
    const emd = logs.length > 0 ? (-finalCritic).toFixed(4) : 0;

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload?.length) {
            return (
                <div className="custom-tooltip">
                    <p className="tooltip-title">Epoch {label}</p>
                    {payload.map((p, i) => (
                        <p key={i} style={{ color: p.color }}>
                            {p.name === 'critic' ? 'Critic Loss' : 'Generator Loss'}: <strong>{p.value}</strong>
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <Section title="Training Metrics" subtitle={`${logs.length} epochs recorded — Critic loss should become more negative, Generator loss should decrease`} id="metrics">
            <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={350}>
                    <LineChart data={logs}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="epoch" stroke="#64748b" minTickGap={25} tickMargin={10} name="Epoch" />
                        <YAxis stroke="#64748b" tickMargin={10} width={50} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="critic" name="Critic Loss" stroke="#ffffff" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="gen" name="Generator Loss" stroke="#6b7280" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="metric-info">
                <div className="info-chip">Final Critic Loss: <strong>{finalCritic}</strong></div>
                <div className="info-chip">Final Generator Loss: <strong>{finalGen}</strong></div>
                <div className="info-chip">Earth Mover Distance (approx): <strong>{emd}</strong></div>
            </div>
        </Section>
    );
}

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

// ─── Main App ───────────────────────────────────────────────────────────────
export default function App() {
    const [activeTab, setActiveTab] = useState('theory');
    const [backendStatus, setBackendStatus] = useState('Checking...');

    useEffect(() => {
        const checkBackend = async () => {
            setBackendStatus('Waking up...');
            try {
                const res = await fetchWithRetry(
                    `${API_BASE_URL}/health`, 
                    {}, 
                    5, 
                    (left) => setBackendStatus(`Retrying... (${left})`)
                );
                if (res.ok) setBackendStatus('Connected');
                else setBackendStatus('Server busy');
            } catch (e) {
                setBackendStatus('Offline');
            }
        };
        checkBackend();
    }, []);

    const tabs = [
        { id: 'theory', label: 'Theory', component: <WGANTheory /> },
        { id: 'generator', label: 'Generator', component: <ImageGenerator /> },
        { id: 'explorer', label: 'Latent Explorer', component: <LatentExplorer /> },
        { id: 'interpolation', label: 'Interpolation', component: <Interpolation /> },
        { id: 'architecture', label: 'Architecture', component: <ArchitectureViewer /> },
        { id: 'metrics', label: 'Metrics', component: <TrainingMetrics /> }
    ];

    return (
        <div className="app-layout">
            <nav className="sidebar">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        className={`sidebar-item ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </nav>
            <main className="main-content">
                <header className="app-header">
                    <h1>Wasserstein <span>GAN</span></h1>
                    <p>Interactive exploration of state-of-the-art generative modeling</p>
                    <div className={`status-pill ${backendStatus.toLowerCase().split(' ')[0].replace('...', '')}`}>
                        Backend: {backendStatus}
                    </div>
                </header>
                <div className="page-content">
                    {tabs.find(t => t.id === activeTab).component}
                </div>
                <Footer />
            </main>
        </div>
    );
}