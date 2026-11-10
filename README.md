# Algorytm Grzeczności

Interaktywna aplikacja edukacyjna (Streamlit + scikit-learn) na stoisko wydziałowe.
Model regresji logistycznej, wytrenowany na 1000 syntetycznych rekordach, przewiduje
prawdopodobieństwo otrzymania prezentu (lub rózgi) na podstawie zachowania ucznia
i pokazuje, które cechy wpłynęły na decyzję.

## Uruchomienie

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Ciemny motyw bazowy jest ustawiony w `.streamlit/config.toml`; font DM Sans i kolory
akcentowe są wstrzykiwane przez CSS w `app.py` (wymaga dostępu do Google Fonts).
