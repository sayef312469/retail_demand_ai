# Retail Demand AI 📊

A comprehensive retail demand forecasting and stock recommendation system combining **Prophet** and **ARIMA** time-series models with Product Viability Index (PVI) scoring and intelligent stock recommendations.

## Features ✨

- 🔮 **Dual Forecasting**: Prophet and ARIMA models with 95% confidence intervals
- 📈 **PVI Scoring**: Demand, Growth, Stability, and Price-based viability analysis (0-100)
- 🎯 **Smart Recommendations**: Increase/Hold/Decrease decisions with confidence levels
- 🚨 **Anomaly Detection**: IQR-based detection of unusual demand patterns
- 📊 **Interactive Dashboard**: Real-time visualization of forecasts and metrics
- 📤 **Upload & Predict**: Upload custom CSV files to get instant forecasts and recommendations
- 🔍 **Model Comparison**: Side-by-side Prophet vs ARIMA comparison

## Tech Stack

**Backend:**
- FastAPI (Python web framework)
- Prophet (Meta's forecasting library)
- ARIMA/Statsmodels (Statistical forecasting)
- Pandas (Data processing)

**Frontend:**
- React (UI framework)
- Recharts (Data visualization)

## Project Structure

```
ai_project/
├── app/
│   └── api.py                 # FastAPI backend
├── src/
│   ├── forecast.py            # Pipeline runner
│   ├── train_prophet.py       # Prophet model training
│   ├── train_arima.py         # ARIMA model training
│   ├── preprocess.py          # Data preprocessing
│   ├── pvi.py                 # PVI score computation
│   ├── recommend.py           # Recommendation engine
│   ├── evaluate.py            # Model evaluation
│   └── plot_reports.py        # Report generation
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── api.js             # API client
│   │   └── components/        # React components
│   └── package.json           # Node dependencies
├── data/
│   ├── raw/                   # Raw input data
│   ├── processed/             # Preprocessed data
│   └── forecast/              # Generated forecasts
└── requirements.txt           # Python dependencies
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (with pip)
- **Node.js 14+** (with npm)
- **Git**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sayef312469/ai_project.git
cd ai_project
```

### 2️⃣ Backend Setup

#### Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## Quick Start

### Option A: Run Full Pipeline + Dashboard

```bash
# 1. Activate Python environment
source venv/bin/activate

# 2. Run the full forecasting pipeline
python src/forecast.py

# 3. Compute PVI scores and recommendations
python src/pvi.py
python src/recommend.py

# 4. Evaluate model accuracy
python src/evaluate.py

# 5. Start FastAPI backend (Terminal 1)
uvicorn app.api:app --reload --port 8000

# 6. Start React frontend (Terminal 2)
cd frontend
npm start
```

The dashboard will open at **http://localhost:3000**

---

### Option B: Run Only API (No Dashboard)

```bash
source venv/bin/activate
uvicorn app.api:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## API Endpoints

### Health & Metadata
- `GET /health` — Check API status
- `GET /summary` — Dashboard summary (PVI, decisions, model performance)

### Stores & Items
- `GET /stores` — List all stores
- `GET /items?store_id=CA_1` — List items for a store

### Forecasts
- `GET /forecast/{store_id}/{item_id}?model=both` — Get Prophet/ARIMA forecasts for an item
- `POST /upload/forecast` — Upload CSV → Get instant forecast + recommendation

### PVI & Recommendations
- `GET /pvi/{store_id}/{item_id}` — Get PVI score for an item
- `GET /recommendations?store_id=CA_1&limit=50` — List stock recommendations

### Example: Upload & Forecast

```bash
curl -X POST "http://localhost:8000/upload/forecast?model=both&periods=6" \
     -F "file=@sales_history.csv"
```

**Required CSV columns:**
- `date` or `week_start_date` — dates (flexible format)
- `sales` or `Weekly_Sales` — sales figures

**Response includes:**
- Prophet forecast with confidence intervals
- ARIMA forecast with model order
- PVI score (0-100) and viability category
- Stock recommendation (Increase/Hold/Decrease)
- Confidence level and explanation

---

## Data Preparation

### Input Data Format

Place raw sales data in `data/raw/`:
- `sales_train_evaluation.csv` — Historical sales
- `calendar.csv` — Date/event information
- `sell_prices.csv` — Product prices

### Run Preprocessing

```bash
python src/preprocess.py
```

This creates `data/processed/processed_m5.csv` with monthly aggregations.

---

## Configuration

### PVI Weights (in `src/pvi.py`)

```python
ALPHA = 0.40   # Demand (40%)
BETA  = 0.25   # Growth (25%)
GAMMA = 0.20   # Stability (20%)
DELTA = 0.15   # Price (15%)
```

### Thresholds (in `src/recommend.py`)

```python
PVI_HIGH   = 67      # High viability threshold
PVI_MEDIUM = 33      # Medium viability threshold
GROWTH_POSITIVE_THRESHOLD = 0.55
ANOMALY_SERIOUS_THRESHOLD = 0.15
```

---

## Model Details

### Prophet
- Seasonal decomposition with yearly & weekly components
- Automatic changepoint detection
- 95% confidence intervals (by default)

### ARIMA
- Automatic order selection via AIC grid search
- Stationarity testing (ADF test)
- p ∈ {0,1,2}, d ∈ {0,1}, q ∈ {0,1,2}
- Confidence intervals via statistical methods

### PVI Scoring

**Formula:**
```
PVI = (0.40 × demand_norm + 0.25 × growth_norm + 0.20 × stability_norm + 0.15 × price_norm) × 100
```

**Sub-scores:**
- **Demand**: Mean forecasted sales (normalized)
- **Growth**: Forecast trend (% change first → last)
- **Stability**: Inverse of coefficient of variation
- **Price**: Average historical sell price

**Categories:**
- High (≥67) — Premium products with strong viability
- Medium (33-67) — Standard products with moderate demand
- Low (<33) — Niche/slow-moving items

### Recommendation Logic

**Layer 1: Anomaly Overrides**
- Serious anomalies + declining demand → **Decrease**
- Serious anomalies + positive trend → **Hold**

**Layer 2: PVI-Driven Matrix**
- High viability + positive growth + low risk → **Increase**
- Low viability + any decline → **Decrease**
- Otherwise → **Hold**

---

## Troubleshooting

### Python Dependencies Issue
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Prophet Not Installing
```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install libffi-dev

# Then reinstall
pip install --no-cache-dir prophet
```

### React Port 3000 Already in Use
```bash
cd frontend
PORT=3001 npm start
```

### API Request Errors

Check that both services are running:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:3000`

Ensure `frontend/package.json` has correct proxy:
```json
"proxy": "http://localhost:8000"
```

---

## Performance Tips

1. **Large Datasets**: Start with `--top-items 100` to test before full pipeline
   ```bash
   python src/forecast.py --top-items 100
   ```

2. **Specific Stores**: Process only target stores
   ```bash
   python src/forecast.py --stores CA_1 TX_1 WI_1
   ```

3. **ARIMA Only** (faster): Skip Prophet for speed
   ```bash
   python src/forecast.py --model arima
   ```

---

## Sample Output

### Dashboard Summary
```
Total Items: 5,420
Total Stores: 10
High Viability: 1,200 (22%)
Medium Viability: 2,800 (52%)
Low Viability: 1,420 (26%)

Recommendations:
  Increase: 1,850
  Hold: 2,890
  Decrease: 680
```

### Individual Recommendation
```
Store: CA_1 | Item: FOODS_1_001
PVI Score: 72.5/100 (High)
Decision: Increase
Confidence: High

Explanation:
PVI=72.5/100 (High viability): demand=0.82, growth=0.68, stability=0.79, price=0.45
Recommend stocking up: positive demand trend, low supply risk, and high viability 
support expansion.
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Authors

**Group B-5**

---

## Support

For issues, questions, or suggestions:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Include error logs and steps to reproduce

---

## Acknowledgments

- Meta's Prophet team for the forecasting library
- Statsmodels community for ARIMA implementations
- M5 Accuracy dataset for training data

