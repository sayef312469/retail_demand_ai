# 🚀 How to Run Retail Demand AI - Step-by-Step Guide for Friends

## Prerequisites ✅

Before starting, make sure you have:
- **Python 3.8 or higher** → [Download here](https://www.python.org/downloads/)
- **Node.js 14 or higher** → [Download here](https://nodejs.org/)
- **Git** → [Download here](https://git-scm.com/)

### Verify Installation
```bash
python --version      # Should show Python 3.8+
node --version        # Should show v14+
npm --version         # Should show npm version
git --version         # Should show git version
```

---

## Step-by-Step Setup ⬇️

### 1. Clone the Repository

Open Terminal/Command Prompt and run:

```bash
git clone https://github.com/sayef312469/ai_project.git
cd ai_project
```

*(Replace `YOUR_USERNAME` with the actual GitHub username)*

---

### 2. Set Up Python Environment

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**You should see `(venv)` at the start of your terminal line now.**

---

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**⏳ This may take 2-5 minutes** (Prophet library is large)

---

### 4. Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

**⏳ This may take 1-2 minutes**

---

## Running the Application 🎯

### Option 1: Full Dashboard (Recommended for first run)

**Keep Python environment activated (`(venv)` should be visible)**

**Terminal 1 - Start Backend API:**
```bash
uvicorn app.api:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
Press CTRL+C to quit
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm start
```

You should see:
```
Compiled successfully!
Local:            http://localhost:3000
```

**Now open http://localhost:3000 in your browser** 🎉

---

### Option 2: Just the API (No Dashboard)

```bash
uvicorn app.api:app --reload --port 8000
```

Then visit **http://localhost:8000/docs** to see interactive API documentation.

---

## Common Tasks 📋

### Upload Your Own Sales Data

1. In the Dashboard, go to the **"Upload"** tab
2. Click to upload a CSV file with columns:
   - `date` or `week_start_date`
   - `sales` or `Weekly_Sales`
3. Select forecasting model (Prophet, ARIMA, or Both)
4. Click "Run forecast"

**Example CSV format:**
```
date,sales
2022-01-01,1500
2022-02-01,1650
2022-03-01,1580
...
```

---

### View Forecasts & Recommendations

1. **Dashboard Tab** - Overall system health
2. **Forecast Tab** - Select store & item to see predictions
3. **PVI Tab** - Viability scores
4. **Recommendations Tab** - Stock decisions
5. **Upload Tab** - Upload and forecast custom data

---

## Troubleshooting 🔧

### Problem: `ModuleNotFoundError: No module named 'prophet'`

**Solution:**
```bash
pip install prophet
```

If still fails, try:
```bash
pip install --no-cache-dir prophet
```

---

### Problem: Port 3000 Already in Use

**Solution:**
```bash
cd frontend
PORT=3001 npm start
```

Then visit http://localhost:3001

---

### Problem: Port 8000 Already in Use

**Solution:**
```bash
uvicorn app.api:app --reload --port 8001
```

Then update frontend proxy in `frontend/package.json` (change 8000 to 8001)

---

### Problem: Git Clone Fails

**Solution:**
```bash
# Make sure you have internet connection
# Try HTTPS instead of SSH:
git clone https://github.com/YOUR_USERNAME/ai_project.git
```

---

### Problem: `npm install` or `pip install` Very Slow

**Solution:** Use a faster mirror or retry later
```bash
# For Python (if in China/Asia):
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# Or:
npm install --registry https://registry.npm.taobao.org
```

---

## First Run Workflow 🌟

1. ✅ Clone repo
2. ✅ Create Python venv
3. ✅ Install Python packages
4. ✅ Install Node packages
5. ✅ Start Backend (Terminal 1)
6. ✅ Start Frontend (Terminal 2)
7. ✅ Open http://localhost:3000
8. ✅ Go to "Upload" tab
9. ✅ Upload a sample CSV
10. ✅ Watch the magic happen! ✨

---

## What Each Section Does

| Page | Purpose |
|------|---------|
| **Dashboard** | Overview stats, model performance, summary |
| **Forecast** | Prophet + ARIMA predictions for items |
| **PVI** | Product viability scores (0-100) |
| **Recommendations** | Stock decisions (Increase/Hold/Decrease) |
| **Charts** | Visualization reports |
| **Upload** | Forecast custom CSV data |
| **Evaluation** | Model accuracy metrics |

---

## API Endpoints (For Advanced Users) 🔌

```bash
# Get API documentation
curl http://localhost:8000/docs

# Get health status
curl http://localhost:8000/health

# List stores
curl http://localhost:8000/stores

# Get forecast for a product
curl "http://localhost:8000/forecast/CA_1/FOODS_1_001?model=both"

# Upload custom CSV
curl -X POST "http://localhost:8000/upload/forecast?model=both" \
     -F "file=@mydata.csv"
```

---

## Still Having Issues? 🆘

1. Check logs for errors
2. Make sure all prerequisites are installed
3. Try restarting both terminals
4. Clear cache: `rm -rf node_modules .cache venv`
5. Reinstall fresh: `npm install` and `pip install -r requirements.txt`

---

## System Requirements

- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 2GB free space
- **Internet**: Required for first setup only
- **Modern Browser**: Chrome, Firefox, Safari, Edge

---

## Performance Notes

- First startup takes ~30-60 seconds while dependencies load
- Prophet fitting takes 5-10 seconds per CSV upload
- ARIMA is typically faster (2-5 seconds)
- Dashboard may take a few seconds to load initial data

---

**Ready to go? Start with Step 1: Clone the Repository** 🚀

For more details, check the main [README.md](README.md)
