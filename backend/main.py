import io, base64, threading, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global training state — polled by frontend /status every ~1.5s
state = {
    "running": False, "epoch": 0, "batch": 0, "total_epochs": 0, "total_batches": 0,
    "g_loss": [], "d_loss": [], "d_x": [], "d_gz": [],
    "images": [],
    "epoch_snapshots": [],
    "done": False, "error": None,
    "config": None,
}
G_model = None
D_model = None
latent_dim_global = 64


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_layers, activation):
        super().__init__()
        act = nn.LeakyReLU(0.2, inplace=True) if activation == "leaky" else nn.ReLU(inplace=True)
        layers, in_dim = [], latent_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), act]
            in_dim = h
        layers += [nn.Linear(in_dim, 784), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, hidden_layers, dropout):
        super().__init__()
        layers, in_dim = [], 784
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, 784))


def tensor_to_b64(img_tensor):
    # img_tensor: (1, 28, 28) in [-1, 1] → PNG base64 at 56×56 nearest-neighbor upscale
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode='L').resize((56, 56), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TrainRequest(BaseModel):
    latent_dim: int = 64
    g_hidden: List[int] = [128, 256]
    d_hidden: List[int] = [512, 256]
    g_activation: str = "leaky"
    dropout: float = 0.2
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.0002
    optimizer: str = "adam"
    label_smoothing: float = 0.1
    sample_interval: int = 50


def train_loop(req: TrainRequest):
    global G_model, D_model, latent_dim_global, state
    try:
        state.update({
            "running": True, "epoch": 0, "batch": 0, "done": False, "error": None,
            "g_loss": [], "d_loss": [], "d_x": [], "d_gz": [],
            "images": [], "epoch_snapshots": [],
            "total_epochs": req.epochs, "config": req.dict(),
        })
        latent_dim_global = req.latent_dim

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        # subset for fast CPU training — full set would be too slow for interactive lab
        if len(dataset) > 12000:
            indices = list(range(12000))
            dataset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(dataset, batch_size=req.batch_size, shuffle=True, drop_last=True)
        state["total_batches"] = len(loader)

        G_model = Generator(req.latent_dim, req.g_hidden, req.g_activation).to(device)
        D_model = Discriminator(req.d_hidden, req.dropout).to(device)

        opt_kwargs = {"betas": (0.5, 0.999)} if req.optimizer == "adam" else {}
        opt_cls = {"adam": optim.Adam, "rmsprop": optim.RMSprop, "sgd": optim.SGD}[req.optimizer]
        opt_G = opt_cls(G_model.parameters(), lr=req.lr, **opt_kwargs)
        opt_D = opt_cls(D_model.parameters(), lr=req.lr, **opt_kwargs)
        bce = nn.BCELoss()

        real_label = 1.0 - req.label_smoothing
        fake_label = 0.0
        step = 0

        for epoch in range(req.epochs):
            state["epoch"] = epoch + 1
            for i, (imgs, _) in enumerate(loader):
                if not state["running"]:
                    return
                imgs = imgs.to(device)
                bs = imgs.size(0)

                # train D
                opt_D.zero_grad()
                real_out = D_model(imgs)
                d_loss_real = bce(real_out, torch.full((bs, 1), real_label, device=device))
                z = torch.randn(bs, req.latent_dim, device=device)
                fake_imgs = G_model(z).detach()
                fake_out = D_model(fake_imgs)
                d_loss_fake = bce(fake_out, torch.full((bs, 1), fake_label, device=device))
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                opt_D.step()

                # train G — non-saturating: fake should be classified as real
                opt_G.zero_grad()
                z = torch.randn(bs, req.latent_dim, device=device)
                fake_imgs = G_model(z)
                g_out = D_model(fake_imgs)
                g_loss = bce(g_out, torch.full((bs, 1), real_label, device=device))
                g_loss.backward()
                opt_G.step()

                step += 1
                state["batch"] = i + 1
                state["g_loss"].append({"step": step, "val": round(g_loss.item(), 4)})
                state["d_loss"].append({"step": step, "val": round(d_loss.item(), 4)})
                state["d_x"].append({"step": step, "val": round(real_out.mean().item(), 4)})
                state["d_gz"].append({"step": step, "val": round(g_out.mean().item(), 4)})

                if step % req.sample_interval == 0:
                    G_model.eval()
                    with torch.no_grad():
                        z_fixed = torch.randn(32, req.latent_dim, device=device)
                        samples = G_model(z_fixed)
                    G_model.train()
                    state["images"] = [tensor_to_b64(samples[k]) for k in range(32)]

            # epoch snapshot for Analyse tab playback
            G_model.eval()
            with torch.no_grad():
                z_snap = torch.randn(8, req.latent_dim, device=device)
                snap_imgs = G_model(z_snap)
            G_model.train()
            state["epoch_snapshots"].append({
                "epoch": epoch + 1,
                "images": [tensor_to_b64(snap_imgs[k]) for k in range(8)],
            })

        state["running"] = False
        state["done"] = True
    except Exception as e:
        state["running"] = False
        state["error"] = str(e)


class GenerateRequest(BaseModel):
    n_samples: int = 16
    temperature: float = 1.0


class InterpolateRequest(BaseModel):
    steps: int = 8


class InspectRequest(BaseModel):
    z: List[float] | None = None


@app.post("/train/start")
def start_training(req: TrainRequest):
    if state["running"]:
        return {"error": "already running"}
    t = threading.Thread(target=train_loop, args=(req,), daemon=True)
    t.start()
    return {"status": "started"}


@app.post("/train/stop")
def stop_training():
    state["running"] = False
    return {"status": "stopped"}


@app.post("/train/reset")
def reset_training():
    global G_model, D_model
    state["running"] = False
    state.update({
        "epoch": 0, "batch": 0, "total_epochs": 0, "total_batches": 0,
        "g_loss": [], "d_loss": [], "d_x": [], "d_gz": [],
        "images": [], "epoch_snapshots": [], "done": False, "error": None, "config": None,
    })
    G_model = None
    D_model = None
    return {"status": "reset"}


@app.get("/status")
def get_status():
    return {
        "running": state["running"], "done": state["done"], "error": state["error"],
        "epoch": state["epoch"], "batch": state["batch"],
        "total_epochs": state["total_epochs"], "total_batches": state["total_batches"],
        "g_loss": state["g_loss"][-300:],
        "d_loss": state["d_loss"][-300:],
        "d_x": state["d_x"][-300:],
        "d_gz": state["d_gz"][-300:],
        "images": state["images"],
        "has_model": G_model is not None,
    }


@app.get("/snapshots")
def get_snapshots():
    return {"snapshots": state["epoch_snapshots"]}


@app.post("/generate")
def generate(req: GenerateRequest):
    if G_model is None:
        return {"error": "no trained model"}
    G_model.eval()
    with torch.no_grad():
        z = torch.randn(req.n_samples, latent_dim_global, device=device) * req.temperature
        imgs = G_model(z)
    G_model.train()
    return {"images": [tensor_to_b64(imgs[k]) for k in range(req.n_samples)]}


@app.post("/interpolate")
def interpolate(req: InterpolateRequest):
    if G_model is None:
        return {"error": "no trained model"}
    G_model.eval()
    with torch.no_grad():
        z_a = torch.randn(1, latent_dim_global, device=device)
        z_b = torch.randn(1, latent_dim_global, device=device)
        alphas = [i / (req.steps - 1) for i in range(req.steps)]
        zs = torch.cat([z_a + t * (z_b - z_a) for t in alphas], dim=0)
        imgs = G_model(zs)
    G_model.train()
    return {"images": [tensor_to_b64(imgs[k]) for k in range(req.steps)]}


@app.post("/inspect")
def inspect(req: InspectRequest):
    if G_model is None:
        return {"error": "no trained model"}
    if req.z is None or len(req.z) != latent_dim_global:
        z = torch.randn(1, latent_dim_global, device=device)
    else:
        z = torch.tensor([req.z], dtype=torch.float32, device=device)
    G_model.eval()
    with torch.no_grad():
        img = G_model(z)
    G_model.train()
    return {"image": tensor_to_b64(img[0]), "z": z.cpu().numpy().tolist()[0], "latent_dim": latent_dim_global}


@app.get("/real-samples")
def real_samples():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    indices = random.sample(range(len(ds)), 8)
    imgs = [tensor_to_b64(ds[i][0]) for i in indices]
    return {"images": imgs}


@app.post("/diversity")
def diversity_score():
    if G_model is None:
        return {"error": "no trained model", "score": 0, "normalized": 0,
                "verdict": "train a model first", "images": []}
    G_model.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim_global, device=device)
        imgs_tensor = G_model(z)
    G_model.train()
    flat = imgs_tensor.view(16, -1).cpu().numpy()
    diffs = []
    for i in range(16):
        for j in range(i + 1, 16):
            diffs.append(float(np.mean(np.abs(flat[i] - flat[j]))))
    score = round(float(np.mean(diffs)), 4)
    verdict = "High diversity — no collapse detected" if score > 0.3 else "Low diversity — possible mode collapse"
    return {
        "score": score,
        "normalized": round(min(score * 2, 1.0), 4),
        "verdict": verdict,
        "images": [tensor_to_b64(imgs_tensor[k]) for k in range(16)],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "has_model": G_model is not None, "latent_dim": latent_dim_global}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
