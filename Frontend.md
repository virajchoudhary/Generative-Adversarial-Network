# Lab Design System
**OpenAI-Aesthetic React + FastAPI Frontend for ATML Labs**
Viraj Choudhary — MPSTME NMIMS Mumbai

Extracted from the working WGAN Lab 5 codebase. Drop these patterns into any new lab and you're 80% done in 10 minutes.

---

## Stack

- **Frontend:** React (Vite/CRA), Recharts for charts
- **Backend:** FastAPI + uvicorn, PyTorch
- **Fonts:** `system-ui` body, `JetBrains Mono` for code/formulas/sliders
- **Key dep:** `import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'`

---

## 1. CSS Variables

```css
:root {
    --bg-primary:    #000000;
    --bg-card:       #111111;
    --bg-card-hover: #141414;
    --bg-input:      rgba(255, 255, 255, 0.04);
    --bg-secondary:  #1a1a1a;
    --border:        rgba(255, 255, 255, 0.08);
    --border-hover:  rgba(255, 255, 255, 0.15);
    --text-primary:  #ffffff;
    --text-secondary:#a3a3a3;
    --text-muted:    #6b7280;
    --radius-sm:     8px;
    --radius-md:     16px;
    --radius-lg:     20px;
    --radius-pill:   999px;
}
```

---

## 2. Page Layout

```css
.app-layout {
    display: flex;
    flex-direction: row;
    min-height: 100vh;
    width: 100%;
    padding: 40px 24px;
    gap: 48px;
    animation: fadeIn 0.6s ease-out;
}
/* mirrors sidebar width on the right so content stays optically centered */
.app-layout::after { content: ''; width: 160px; flex-shrink: 0; }

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 0;
    min-height: 100%;
}

.page-content {
    flex: 1 0 auto;
    min-height: 420px;
}

.main-content > * { width: 100%; max-width: 1100px; }

.main-content .app-header { text-align: center; margin-bottom: 40px; }

.app-header h1 {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--text-primary);
}

.app-header p {
    color: var(--text-secondary);
    font-size: 1.15rem;
    max-width: 600px;
    margin: 0 auto;
}
```

---

## 3. Sidebar Navigation

```css
.sidebar {
    width: 160px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding-top: 8px;
}

.sidebar-item {
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    padding: 8px 12px;
    text-align: left;
    font-size: 14px;
    color: #6b7280;
    cursor: pointer;
    border-radius: 0 6px 6px 0;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
    font-family: inherit;
}

.sidebar-item:hover { color: #d1d5db; background: rgba(255,255,255,0.04); }

.sidebar-item.active {
    border-left: 2px solid #ffffff;
    color: #ffffff;
    font-weight: 500;
    background: rgba(255,255,255,0.05);
}
```

JSX pattern -- copy this for every lab:

```jsx
export default function App() {
    const [activeTab, setActiveTab] = useState('theory');

    const tabs = [
        { id: 'theory',    label: 'Theory',    component: <TheorySection /> },
        { id: 'generator', label: 'Generator', component: <GeneratorSection /> },
        { id: 'metrics',   label: 'Metrics',   component: <MetricsSection /> },
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
                    <h1>Lab Title</h1>
                    <p>One line description</p>
                </header>
                <div className="page-content">
                    {tabs.find(t => t.id === activeTab).component}
                </div>
                <Footer />
            </main>
        </div>
    );
}
```

---

## 4. Section Component

Every tab's content lives in a Section. All sections are always visible -- no accordions.

```jsx
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
```

```css
.card {
    background: var(--bg-card);
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    margin-bottom: 32px;
    overflow: hidden;
}
.card-header { padding: 32px 32px 24px; }
.card-header h2 { margin: 0; font-size: 1.4rem; font-weight: 700; color: var(--text-primary); }
.card-subtitle { margin: 6px 0 0; font-size: 1rem; color: var(--text-secondary); }
.card-body { padding: 0 32px 32px; }
```

---

## 5. Buttons

```css
.btn-primary {
    padding: 12px 28px;
    border: none;
    border-radius: var(--radius-pill);
    background: #ffffff;
    color: #000000;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    font-family: inherit;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.btn-primary:hover:not(:disabled) { background: #e5e5e5; }
.btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }

.btn-sm {
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-weight: 500;
    font-size: 0.85rem;
    cursor: pointer;
    font-family: inherit;
}
.btn-sm:hover { background: rgba(255,255,255,0.1); }
.btn-outline { background: transparent; color: var(--text-secondary); }

.spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(0,0,0,0.1);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
}
.spinner-light {
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #fff;
}
@keyframes spin { to { transform: rotate(360deg); } }
```

Usage:
```jsx
<button className="btn-primary" onClick={run} disabled={loading}>
    {loading ? <span className="spinner"></span> : null}
    {loading ? 'Running...' : 'Run'}
</button>
```

---

## 6. Sliders and Controls

```css
.controls-row { display: flex; align-items: flex-end; gap: 32px; margin-bottom: 32px; flex-wrap: wrap; }
.control-group { flex: 1; min-width: 200px; }
.control-group label { display: block; font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px; }
.control-group label strong { color: var(--text-primary); font-weight: 600; }
.range-labels { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); margin-top: 8px; }

input[type=range] {
    -webkit-appearance: none;
    width: 100%; height: 4px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 16px; width: 16px;
    border-radius: 50%;
    background: var(--text-primary);
    cursor: pointer;
}
```

---

## 7. Code Blocks and Formulas

```css
.formula {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--text-primary);
    margin: 16px 0;
    line-height: 1.8;
    overflow-x: auto;
}

.algorithm {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 24px 32px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 2;
}
.algorithm strong { color: var(--text-primary); }
.indent1 { padding-left: 24px; }
.indent2 { padding-left: 48px; }
.algorithm p { margin: 0; }
```

Algorithm JSX pattern:
```jsx
<div className="algorithm">
    <p><strong>for</strong> each iteration:</p>
    <p className="indent1"><strong>for</strong> t = 1, ..., n:</p>
    <p className="indent2">Compute loss</p>
    <p className="indent2">Update: θ ← θ − α·∇L</p>
</div>
```

---

## 8. Theory Grid

```css
.theory-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

.theory-item {
    background: var(--bg-input);
    border-radius: var(--radius-sm);
    padding: 32px;
    border: 1px solid var(--border);
}
.theory-item.full-width { grid-column: 1 / -1; }
.theory-item h4 { margin: 0 0 16px; font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
.theory-item p { color: var(--text-secondary); line-height: 1.7; }
```

---

## 9. Comparison Tables

```css
.theory-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.theory-table th {
    text-align: left; padding: 16px 20px;
    color: #ffffff; font-weight: 600;
    border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
}
.theory-table td { padding: 16px 20px; border-bottom: 1px solid var(--border); color: #a3a3a3; }
```

---

## 10. Architecture / Layer Badges

```css
.type-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    font-size: 0.75rem; font-weight: 500;
    background: rgba(255,255,255,0.08);
    color: var(--text-primary);
}

.config-chip {
    padding: 8px 16px;
    border-radius: var(--radius-pill);
    background: var(--bg-input);
    border: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}
```

All layer type badges are uniform gray -- no color coding by layer type.

---

## 11. Training Metrics Chart

```jsx
<ResponsiveContainer width="100%" height={350}>
    <LineChart data={logs}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="epoch" stroke="#64748b" minTickGap={25} tickMargin={10} />
        <YAxis stroke="#64748b" tickMargin={10} width={50} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Line type="monotone" dataKey="loss_a" stroke="#ffffff" strokeWidth={2.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="loss_b" stroke="#6b7280" strokeWidth={2.5} dot={false} isAnimationActive={false} />
    </LineChart>
</ResponsiveContainer>
```

Rule: first metric white (`#ffffff`), second metric muted gray (`#6b7280`). No colored lines.

---

## 12. Debounced Slider -> API Pattern

Wrap in `useRef` to prevent recreation on every render -- this is the pattern that actually works:

```jsx
import { useRef } from 'react';

function debounce(func, wait) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    };
}

function MyComponent() {
    const abortRef = useRef(null);

    const debouncedCall = useRef(debounce(async (payload) => {
        if (abortRef.current) abortRef.current.abort();
        abortRef.current = new AbortController();
        try {
            const res = await fetch('http://127.0.0.1:8000/your-endpoint', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortRef.current.signal
            });
            const data = await res.json();
            setResult(data);
        } catch (e) {
            if (e.name !== 'AbortError') console.error(e);
        }
    }, 300)).current;

    // inside slider onChange:
    // onChange={(e) => { updateState(e.target.value); debouncedCall(newPayload); }}
}
```

The `AbortController` cancels stale in-flight requests when the slider moves again before the previous response arrives.

---

## 13. FastAPI Backend Skeleton

```python
import io, base64, torch, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_base64(tensor):
    img = (tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    buf = io.BytesIO()
    transforms.ToPILImage()(img.cpu()).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

class GenerateRequest(BaseModel):
    num_samples: int = 16

@app.post("/generate")
def generate(req: GenerateRequest):
    # your model inference here
    return { "images": [...], "scores": [...] }

@app.get("/logs")
def get_logs():
    # return loss_log.json contents
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 14. Text Inputs

```css
.text-input {
    width: 100%;
    padding: 14px 18px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.2s;
    resize: none;
}

.text-input:focus { border-color: var(--border-hover); }
.text-input::placeholder { color: var(--text-muted); }

textarea.text-input {
    min-height: 120px;
    line-height: 1.6;
}

.input-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 24px;
}
.input-row .text-input { flex: 1; }
```

---

## 15. Footer Component

```css
.app-footer {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 40px;
    padding: 32px 0 20px;
    margin-top: auto;
    border-top: 1px solid var(--border);
}

.footer-links { display: flex; gap: 32px; }

.footer-links a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.88rem;
    transition: color 0.15s;
}

.footer-links a:hover { color: var(--text-primary); }
.footer-copy { color: var(--text-muted); font-size: 0.88rem; }
```

```jsx
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
```

---

## 16. New Lab Checklist

1. Copy `App.css` from WGAN lab. Delete component-specific rules (image grid, interp strip, explorer layout). Keep all variables and base styles.
2. Copy the `Section` component and `debounce` function verbatim.
3. Set `const API = 'http://127.0.0.1:8000'` at the top of `App.js` -- never use `localhost`.
4. Define `tabs` array, wire to sidebar and `activeTab` state.
5. Build one component per tab, wrap everything in `<Section title="..." subtitle="...">`.
6. Theory content: `.theory-grid` + `.theory-item` + `.formula` + `.algorithm`.
7. Any slider calling API: use the `useRef(debounce(...)).current` pattern from section 12.
8. Comparison tables: `.theory-table` with white headers, gray data rows.
9. Charts: white line for primary metric, `#6b7280` for secondary. No colored lines.
10. All sections always visible -- no collapse/accordion toggles anywhere.
11. Backend: `host="0.0.0.0"` in uvicorn. Images returned as `data:image/png;base64,...` strings.
12. Image rendering for ML outputs: always add `image-rendering: pixelated` to prevent blurry upscaling.

---
