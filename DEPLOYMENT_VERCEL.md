# Deploying Smart E-Waste Analyzer to Vercel

This repository is configured for one-click deployment on **Vercel** with a decoupled serverless architecture.

---

## 1. Project Architecture on Vercel

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend Web App** | Modern HTML5/CSS3/JavaScript (`public/`) | Responsive interactive dashboard with Canvas YOLO bounding box visualizer, INR salvage valuation breakdown, and recycling safety guide. |
| **Serverless API** | Python Serverless Function (`api/index.py`) | Microservice providing taxonomy, pricing benchmarks (INR), hazard profiles, and API health endpoints. |
| **Deep Learning Backend (Heavy)** | Streamlit / FastAPI (`app.py` + `yolov8s_pcb_best.pt`) | Full-precision YOLOv8s inference engine (PyTorch). Best hosted on **Streamlit Community Cloud**, **Hugging Face Spaces**, or **Render**. |

> **Why decoupled?**  
> Vercel Serverless Functions enforce a **250 MB uncompressed bundle limit**. The full PyTorch suite + OpenCV + CUDA binaries + the 89.5 MB YOLOv8s weights exceed 600 MB. Decoupling the frontend on Vercel from the ML backend ensures fast CDN loading, global availability, and avoids serverless memory/timeout errors.

---

## 2. Deploy to Vercel via Web (GitHub)

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Configure Vercel deployment and serverless API"
   git push origin main
   ```
2. Navigate to [vercel.com](https://vercel.com) and log in.
3. Click **Add New** &rarr; **Project**.
4. Import your `PCB-Component-Detection` repository.
5. In the project settings:
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: Leave default (empty)
   - **Output Directory**: Leave default (uses `vercel.json` and `public/`)
6. Click **Deploy**.

---

## 3. Deploy via Vercel CLI

You can deploy directly from your local terminal:

```bash
# 1. Install Vercel CLI (if not already installed)
npm install -g vercel

# 2. Log in to your Vercel account
vercel login

# 3. Deploy preview
vercel

# 4. Deploy to production
vercel --prod
```

---

## 4. Hosting the Full ML Model Backend (Free Options)

If you want the full PyTorch YOLOv8s model live with live camera/uploaded inference, pair your Vercel app with one of these free GPU/CPU hosting platforms:

1. **Streamlit Community Cloud (Recommended & 100% Free)**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Select your repo and branch.
   - Set **Main file path** to `app.py`.
   - Click **Deploy**.

2. **Hugging Face Spaces (Free CPU & 50GB storage)**:
   - Create a new Space with the `Streamlit` SDK.
   - Push this repository to the Space.

3. **Render (Free tier, auto-sleeps after 15 min idle)**:
   - Create a new **Web Service**, connect your repo.
   - Set **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 5. Troubleshooting Common Streamlit Cloud Errors

### `ImportError: Ultralytics library is not installed`
This is almost always caused by `opencv-python` (the GUI build) failing to import on headless Linux servers. Streamlit Community Cloud runs Ubuntu without a display server, so `libGL.so.1` is missing.

**Fix (already applied in this repo):**
- `requirements.txt` uses `opencv-python-headless` instead of `opencv-python`.
- `packages.txt` installs `libgl1-mesa-glx` and `libglib2.0-0` as system-level dependencies.

If you still see the error after redeploying, click **Manage app → Reboot app** in the bottom-right of your Streamlit Cloud dashboard to force a clean reinstall of all packages.

### `FileNotFoundError: Model weights file not found`
The YOLOv8s checkpoint (`models/yolov8s_pcb_best.pt`, ~89 MB) must be pushed to the repository. Verify it is **not** excluded by `.gitignore` and has been committed:
```bash
git lfs track "models/*.pt"
git add models/yolov8s_pcb_best.pt
git commit -m "add: YOLOv8s PCB model weights"
git push
```
> If the file is >100 MB, use [Git LFS](https://git-lfs.github.com/) (free on GitHub).

### App is very slow on first load
The YOLOv8s model is cached via `@st.cache_resource`. The first request after a cold start triggers model loading (~5–15 seconds). Subsequent requests are fast. This is expected behaviour on free-tier CPU hosts.
