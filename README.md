# StockApp (Starter)
Privacy-first paper trading prototype using Python + Streamlit with `yfinance` (no API key).

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Uses replay mode over recent intraday candles (1m) so you can step through ticks.
- Fractional shares supported. No real trading. Educational only.
