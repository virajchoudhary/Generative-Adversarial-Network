# Generative Adversarial Network

A FastAPI service and React/Vite client for interactive GAN training and image-generation demos.

## What it does

`backend/main.py` defines a generator and discriminator and exposes routes to start, stop, and reset training; read status and snapshots; generate and interpolate images; inspect model output; and request sample or diversity data.

Training and sample routes use Fashion-MNIST. Model quality, training convergence, and diversity scores have not been independently validated.

## Run the API

From `backend/` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --reload
```

On macOS or Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

From `frontend/`, run `npm ci`, then `npm run dev`. Development defaults to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` to override it.

The loader can download Fashion-MNIST on the first training or sample request. Running the API from `backend/` makes its `./data` path relative to that directory.

The commands match the source configuration. Installation and runtime behavior were not retested in this documentation pass.

## Data source

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is a Zalando Research benchmark. Its upstream repository provides an [MIT notice](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE). Preserve that notice with redistributed upstream material. This attribution does not assign a license to this application's code.
