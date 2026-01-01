# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
cd d:\soham\crisis_forecasting
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py --all
```

This will:
- ✓ Generate 847 synthetic data records per domain
- ✓ Process data (clean, align, fuse, engineer 200+ features)
- ✓ Train ensemble ML models (RF + XGBoost + LSTM)
- ✓ Generate predictions and risk scores
- ✓ Run Monte Carlo scenario simulations
- ✓ Create crisis alerts

**Estimated time**: 5-10 minutes

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 What You'll See

### Dashboard Pages

1. **Home** - Overview with key metrics
2. **Global Overview** - Interactive world map with crisis hotspots
3. **Risk Analysis** - Trends, correlations, comparisons
4. **Forecasts** - 12-month predictions with confidence intervals
5. **Scenarios** - Best/expected/worst case simulations
6. **Alerts** - Real-time crisis warnings
7. **About** - Full methodology, ethics, technical details

---

## 🎯 Project Highlights

✅ **30+ Python modules** implementing PhD-level data science  
✅ **Multi-domain fusion** (climate + health + food + economic)  
✅ **200+ engineered features** (rolling stats, lags, interactions)  
✅ **Ensemble ML** (Random Forest + XGBoost + LSTM)  
✅ **Monte Carlo simulations** for scenario planning  
✅ **Interactive dashboard** with 6 pages of visualizations  
✅ **Comprehensive ethics** and methodology documentation  

---

## 📁 Key Files

- `main.py` - Complete pipeline CLI
- `requirements.txt` - All dependencies
- `README.md` - Full documentation
- `dashboard/app.py` - Streamlit dashboard
- `config/config.yaml` - System configuration

---

## 💡 CLI Options

```bash
# Individual steps
python main.py --collect-data    # Generate data
python main.py --preprocess      # Clean & engineer features
python main.py --train-models    # Train ensemble
python main.py --predict         # Generate forecasts
python main.py --scenarios       # Run simulations

# Or run everything
python main.py --all

# Launch dashboard
python main.py --dashboard
```

---

## 🎓 This Project Demonstrates

- ✅ PhD-level systems thinking
- ✅ Production ML engineering
- ✅ Multi-domain data fusion
- ✅ Ensemble learning techniques
- ✅ Risk assessment frameworks
- ✅ Interactive visualizations
- ✅ Ethical AI principles
- ✅ Comprehensive documentation

**Perfect for**: Resumes, portfolios, interviews, academic papers, or real-world adaptation!

---

**Ready to impress? This is PhD-level work!** 🚀
