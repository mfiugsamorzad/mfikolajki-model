"""
MLikołaj
========

ML = machine learning (uczenie maszynowe) + Mikołaj.

Interaktywna aplikacja edukacyjna (Streamlit + scikit-learn) na stoisko wydziałowe.

Co się tu dzieje?
1. Generujemy syntetyczny (zmyślony, ale logiczny) zbiór danych o 1000 uczniach.
2. Trenujemy na nim regresję logistyczną – prosty i przejrzysty model klasyfikacyjny.
3. Użytkownik ustawia swoje „zachowanie” suwakami i przyciskami,
   a model na bieżąco liczy prawdopodobieństwo otrzymania prezentu (klasa 1)
   zamiast rózgi (klasa 0).
4. Wykresy pokazują, JAK model podjął decyzję – czyli wagi poszczególnych cech.

Autor: Rada Samorządu Studenckiego WMFI UG

Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. KONFIGURACJA STRONY I PALETA KOLORÓW
# ---------------------------------------------------------------------------

# Konfiguracja strony musi być pierwszym wywołaniem Streamlita w skrypcie.
st.set_page_config(
    page_title="MLikołaj",
    page_icon="🎄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paleta kolorów – świąteczna: ciemna zieleń świerku, bordo i złoto.
# Trzymamy ją w jednym miejscu, żeby UI i wykresy były spójne.
KOLOR_TLO = "#0A1C15"        # tło całej aplikacji (ciemny świerk)
KOLOR_TLO_2 = "#12352A"      # jaśniejsza zieleń do delikatnej poświaty u góry strony
KOLOR_PANEL = "#10271E"      # tło „kart” (paneli)
KOLOR_RAMKA = "#24463A"      # delikatne obramowania i siatka wykresów
KOLOR_TEKST = "#F4EEE2"      # główny tekst (ciepła, kremowa biel)
KOLOR_TEKST_2 = "#A7B8AD"    # tekst pomocniczy (opisy, podpisy osi)
KOLOR_ZIELONY = "#5CBF80"    # „prezent” / wpływ dodatni (jasna zieleń choinki)
KOLOR_BORDO = "#C2405A"      # „rózga” / wpływ ujemny (bordo, rozjaśnione dla czytelności)
KOLOR_BORDO_CIEMNE = "#7A1F33"  # głębokie bordo do wstążek i hover
KOLOR_ZLOTY = "#E2B04A"      # akcent interfejsu (suwaki, przyciski) i wynik „na granicy”

# Ziarno losowości – dzięki niemu dane i model są za każdym razem identyczne.
ZIARNO = 42
LICZBA_REKORDOW = 1000

# Proste, liniowe ikony SVG (kolor dziedziczą z CSS przez „currentColor”).
IKONA_GWIAZDA = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linejoin="round"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 '
    '2.9 1-6.1-4.4-4.3 6.1-.9z"/></svg>'
)
IKONA_PREZENT = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="18" '
    'height="4" rx="1"/><path d="M5 12v8a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-8M12 8v13"/>'
    '<path d="M12 8C10.5 4.5 7 4 7 6.5S10 8 12 8zM12 8c1.5-3.5 5-4 5-1.5S14 8 12 8z"/></svg>'
)
IKONA_ROZGA = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round"><path d="M12 21V4M12 21L6.5 6M12 21l5.5-15M12 13l-2.5-3'
    'M12 9l2.5-2.5M8.5 11.5 6.5 11M15.6 11.5l2-.8"/><path d="M9.6 17.5h4.8"/></svg>'
)

# Autor aplikacji i ścieżka do logo (białe logo na przezroczystym tle – pasuje do ciemnego motywu)
AUTOR = "Rada Samorządu Studenckiego WMFI UG"
SCIEZKA_LOGO = Path(__file__).parent / "assets" / "logo_rss.png"

# Opis cech: klucz techniczny -> czytelna nazwa wyświetlana w interfejsie.
CECHY = {
    "godziny_gier": "Granie na komputerze [h/tydz.]",
    "odpisane_zadania": "Odpisane zadania domowe",
    "porcje_warzyw": "Porcje warzyw [tydz.]",
    "sprzatanie": "Sprzątanie bez przypominania",
    "kolka_naukowe": "Kółko mat./inf.",
}


# ---------------------------------------------------------------------------
# 2. STYL – CIEMNY MOTYW I FONT „DM SANS”
# ---------------------------------------------------------------------------

def wstrzyknij_css() -> None:
    """Wstrzykuje własny CSS: font DM Sans, ciemny motyw i kolory akcentowe."""
    st.markdown(
        f"""
        <style>
        /* Pobranie fontu DM Sans z Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap');

        /* Wymuszenie DM Sans dla całej aplikacji.
           Wyjątek: ikony Streamlita (Material Symbols) – inaczej zamieniłyby się w tekst. */
        html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
            font-family: 'DM Sans', sans-serif !important;
        }}

        /* Ciemne, świerkowe tło z delikatną poświatą u góry */
        .stApp {{
            background: radial-gradient(ellipse 90% 55% at 50% 0%, {KOLOR_TLO_2} 0%, {KOLOR_TLO} 70%);
            background-attachment: fixed;
            color: {KOLOR_TEKST};
        }}

        /* Padający śnieg – czysty CSS: kilka warstw drobnych kropek przesuwanych w dół.
           pointer-events: none sprawia, że śnieg nie blokuje kliknięć w suwaki. */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 999;
            opacity: 0.5;
            background-image:
                radial-gradient(2.6px 2.6px at 20px 30px, #fff 50%, transparent 51%),
                radial-gradient(2px 2px at 140px 90px, #fff 50%, transparent 51%),
                radial-gradient(3px 3px at 260px 160px, #fff 50%, transparent 51%),
                radial-gradient(2.2px 2.2px at 80px 220px, #fff 50%, transparent 51%),
                radial-gradient(1.8px 1.8px at 200px 260px, #fff 50%, transparent 51%);
            background-size: 300px 300px;
            animation: snieg 22s linear infinite;
        }}
        .stApp::after {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 999;
            opacity: 0.3;
            background-image:
                radial-gradient(4px 4px at 60px 50px, #fff 50%, transparent 51%),
                radial-gradient(3.2px 3.2px at 330px 210px, #fff 50%, transparent 51%),
                radial-gradient(3.6px 3.6px at 190px 380px, #fff 50%, transparent 51%);
            background-size: 420px 420px;
            animation: snieg-wolny 34s linear infinite;
        }}
        @keyframes snieg {{
            from {{ background-position: 0 0; }}
            to   {{ background-position: 60px 300px; }}
        }}
        @keyframes snieg-wolny {{
            from {{ background-position: 0 0; }}
            to   {{ background-position: -80px 420px; }}
        }}
        /* Osoby z włączonym ograniczeniem animacji w systemie – śnieg stoi w miejscu */
        @media (prefers-reduced-motion: reduce) {{
            .stApp::before, .stApp::after {{ animation: none; }}
        }}

        /* Ukrycie domyślnego menu i stopki – czystszy wygląd na stoisku */
        #MainMenu, footer, header {{ visibility: hidden; }}

        /* Nagłówki: mocna waga, lekko ściśnięte litery */
        h1, h2, h3, h4 {{
            color: {KOLOR_TEKST} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        /* Mały, techniczny nadtytuł nad nagłówkiem */
        .nadtytul {{
            color: {KOLOR_ZLOTY};
            font-size: 0.8rem;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: -0.4rem;
        }}

        /* Tytuł z gwiazdką (jak na czubku choinki) */
        .tytul {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {KOLOR_TEKST};
            margin: 0.2rem 0 0.4rem 0;
            line-height: 1.1;
        }}
        .tytul svg {{
            width: 2.4rem;
            height: 2.4rem;
            color: {KOLOR_ZLOTY};
            filter: drop-shadow(0 0 10px rgba(226, 176, 74, 0.45));
        }}

        /* Ozdobny separator: złota linia z gwiazdką pośrodku */
        .ornament {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 1.4rem 0 1.6rem 0;
            color: {KOLOR_ZLOTY};
        }}
        .ornament::before, .ornament::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, transparent, {KOLOR_RAMKA} 15%, {KOLOR_ZLOTY}88 50%, {KOLOR_RAMKA} 85%, transparent);
        }}
        .ornament svg {{ width: 1.1rem; height: 1.1rem; }}

        /* Stopka z życzeniami */
        .stopka {{
            text-align: center;
            color: {KOLOR_TEKST_2};
            font-size: 0.95rem;
            letter-spacing: 0.04em;
            padding-bottom: 1rem;
        }}
        .stopka b {{ color: {KOLOR_ZLOTY}; font-weight: 500; }}

        /* Nagłówek: tekst po lewej, małe logo autora po prawej */
        .naglowek {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 2rem;
        }}
        .logo-naglowek {{
            height: 84px;
            width: auto;
            opacity: 0.9;
            margin-top: 0.4rem;
        }}
        /* Na wąskich ekranach logo w nagłówku znika (zostaje w stopce) */
        @media (max-width: 760px) {{
            .logo-naglowek {{ display: none; }}
        }}

        /* Stopka: małe logo i podpis autora */
        .autor {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.8rem;
            color: {KOLOR_TEKST_2};
            font-size: 0.85rem;
            margin-top: -0.4rem;
            padding-bottom: 1.2rem;
        }}
        .logo-stopka {{ height: 36px; width: auto; opacity: 0.75; }}

        .podtytul {{
            color: {KOLOR_TEKST_2};
            font-size: 1.05rem;
            max-width: 52rem;
        }}

        /* Karta z dużym wynikiem procentowym */
        /* Karta wyniku ozdobiona jak paczka: bordowa wstążka z złotym brzegiem */
        .karta-wyniku {{
            position: relative;
            overflow: hidden;
            background: {KOLOR_PANEL};
            border: 1px solid {KOLOR_RAMKA};
            border-radius: 14px;
            padding: 1.4rem 1.6rem 1.1rem 1.6rem;
            margin-bottom: 0.8rem;
        }}
        .karta-wyniku::before {{
            content: "";
            position: absolute;
            top: 0; bottom: 0;
            right: calc(2.9rem - 8px);  /* środek wstążki = środek ikony */
            width: 16px;
            background: {KOLOR_BORDO_CIEMNE};
            border-left: 1px solid {KOLOR_ZLOTY}66;
            border-right: 1px solid {KOLOR_ZLOTY}66;
        }}
        .karta-wyniku .ikona {{
            position: absolute;
            top: 1.2rem; right: 1.4rem;
            width: 3rem; height: 3rem;
            padding: 0.45rem;
            box-sizing: border-box;
            background: {KOLOR_PANEL};
            border: 1px solid {KOLOR_ZLOTY}88;
            border-radius: 50%;
        }}
        .karta-wyniku .ikona svg {{ width: 100%; height: 100%; display: block; }}
        .karta-wyniku .etykieta {{
            color: {KOLOR_TEKST_2};
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .karta-wyniku .liczba {{
            font-size: 4.2rem;
            font-weight: 700;
            line-height: 1.05;
            font-variant-numeric: tabular-nums;
        }}
        .karta-wyniku .werdykt {{
            font-size: 1.15rem;
            font-weight: 500;
        }}

        /* Suwaki – złoty akcent */
        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {KOLOR_ZLOTY} !important;
            border-color: {KOLOR_ZLOTY} !important;
            box-shadow: none !important;
        }}

        /* Etykiety kontrolek nieco jaśniejsze i grubsze */
        .stSlider label p, .stRadio > label p {{
            color: {KOLOR_TEKST} !important;
            font-weight: 500 !important;
        }}

        /* Przyciski (np. „Reset”) – obramowanie w kolorze akcentu */
        .stButton > button {{
            background: transparent;
            color: {KOLOR_ZLOTY};
            border: 1px solid {KOLOR_ZLOTY};
            border-radius: 8px;
            font-weight: 500;
        }}
        .stButton > button:hover, .stButton > button:focus {{
            background: {KOLOR_BORDO_CIEMNE};
            color: {KOLOR_TEKST};
            border-color: {KOLOR_ZLOTY};
        }}

        /* Pasek postępu – grubszy, zaokrąglony, na ciemnym torze */
        div[data-testid="stProgress"] div[role="progressbar"] > div {{
            height: 14px !important;
            border-radius: 7px !important;
            background-color: {KOLOR_RAMKA} !important;
        }}
        div[data-testid="stProgress"] div[role="progressbar"] > div > div {{
            border-radius: 7px !important;
        }}

        /* Ramka panelu wejściowego (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {KOLOR_PANEL};
            border-color: {KOLOR_RAMKA} !important;
            border-radius: 14px;
        }}

        /* Pole kodu (np. `scikit-learn`) w zieleni zamiast domyślnego szarego */
        .stMarkdown code {{
            background: {KOLOR_RAMKA};
            color: {KOLOR_ZLOTY};
        }}

        /* Cienka linia oddzielająca sekcje */
        hr {{ border-color: {KOLOR_RAMKA} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def logo_base64() -> str:
    """Zwraca logo jako tekst base64 (do osadzenia w HTML) albo pusty tekst, gdy pliku brak."""
    if not SCIEZKA_LOGO.exists():
        return ""
    return base64.b64encode(SCIEZKA_LOGO.read_bytes()).decode()


def logo_html(klasa: str) -> str:
    """Znacznik <img> z logo autora (pusty, jeśli pliku z logo nie ma)."""
    dane = logo_base64()
    if not dane:
        return ""
    return f'<img class="{klasa}" src="data:image/png;base64,{dane}" alt="Logo: {AUTOR}">'


def ornament() -> None:
    """Rysuje ozdobny separator sekcji (złota linia z gwiazdką)."""
    st.markdown(f'<div class="ornament">{IKONA_GWIAZDA}</div>', unsafe_allow_html=True)


def pokoloruj_pasek_postepu(kolor: str) -> None:
    """Dynamicznie zmienia kolor wypełnienia paska postępu zależnie od wyniku."""
    st.markdown(
        f"""
        <style>
        div[data-testid="stProgress"] div[role="progressbar"] > div > div {{
            background: {kolor} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 3. DANE SYNTETYCZNE
# ---------------------------------------------------------------------------

@st.cache_data
def wygeneruj_dane(n: int = LICZBA_REKORDOW, ziarno: int = ZIARNO) -> pd.DataFrame:
    """
    Tworzy syntetyczny zbiór danych o uczniach.

    Najpierw losujemy cechy, a potem – według ustalonej przez nas „prawdziwej”
    reguły – liczymy szansę na prezent i losujemy etykietę (0 = rózga, 1 = prezent).
    Model NIE zna tej reguły – musi ją odkryć sam na podstawie przykładów.
    """
    rng = np.random.default_rng(ziarno)

    # Losowanie cech w zakresach zgodnych z suwakami w interfejsie
    godziny_gier = rng.integers(0, 41, n)        # 0–40 h tygodniowo
    odpisane_zadania = rng.integers(0, 21, n)    # 0–20 zadań
    porcje_warzyw = rng.integers(0, 31, n)       # 0–30 porcji
    sprzatanie = rng.integers(0, 2, n)           # 0 = Nie, 1 = Tak
    kolka_naukowe = rng.binomial(1, 0.35, n)     # ~35% uczniów chodzi na kółka

    # „Ukryta” reguła świata (logit). Znaki wag są celowo logiczne:
    #   + warzywa, + sprzątanie, + kółka naukowe  -> zwiększają szansę na prezent
    #   - odpisywanie zadań, - dużo grania        -> zmniejszają szansę
    logit = (
        1.0
        - 0.05 * godziny_gier
        - 0.30 * odpisane_zadania
        + 0.12 * porcje_warzyw
        + 1.00 * sprzatanie
        + 1.40 * kolka_naukowe
    )

    # Funkcja sigmoidalna zamienia logit na prawdopodobieństwo z przedziału (0, 1)
    p_prezent = 1.0 / (1.0 + np.exp(-logit))

    # Losujemy etykietę – dzięki temu dane mają realistyczny „szum”
    prezent = rng.binomial(1, p_prezent)

    return pd.DataFrame(
        {
            "godziny_gier": godziny_gier,
            "odpisane_zadania": odpisane_zadania,
            "porcje_warzyw": porcje_warzyw,
            "sprzatanie": sprzatanie,
            "kolka_naukowe": kolka_naukowe,
            "prezent": prezent,
        }
    )


# ---------------------------------------------------------------------------
# 4. TRENING MODELU
# ---------------------------------------------------------------------------

@st.cache_resource
def wytrenuj_model(dane: pd.DataFrame):
    """
    Trenuje regresję logistyczną i zwraca (model, dokładność na zbiorze testowym).

    StandardScaler sprowadza cechy do wspólnej skali (średnia 0, odchylenie 1).
    Dzięki temu współczynniki modelu można uczciwie porównywać między sobą –
    np. „godziny” i „Tak/Nie” mają zupełnie różne zakresy.
    """
    X = dane[list(CECHY)]
    y = dane["prezent"]

    # Podział: 80% danych do nauki, 20% do sprawdzenia, czy model „zdał egzamin”
    X_tren, X_test, y_tren, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ZIARNO, stratify=y
    )

    model = Pipeline(
        [
            ("skaler", StandardScaler()),
            ("regresja", LogisticRegression()),
        ]
    )
    model.fit(X_tren, y_tren)

    dokladnosc = accuracy_score(y_test, model.predict(X_test))
    return model, dokladnosc


# ---------------------------------------------------------------------------
# 5. WYKRESY (matplotlib w ciemnym motywie)
# ---------------------------------------------------------------------------

# Matplotlib nie korzysta z Google Fonts – sprawdzamy, czy DM Sans jest w systemie,
# żeby uniknąć ostrzeżeń „findfont” w konsoli.
FONT_WYKRESOW = (
    "DM Sans"
    if any(f.name == "DM Sans" for f in font_manager.fontManager.ttflist)
    else "sans-serif"
)

def przygotuj_os(fig, ax) -> None:
    """Wspólne, ciemne ostylowanie wykresu: tło, osie, siatka, font."""
    fig.patch.set_facecolor(KOLOR_PANEL)
    ax.set_facecolor(KOLOR_PANEL)
    for strona in ("top", "right", "left"):
        ax.spines[strona].set_visible(False)
    ax.spines["bottom"].set_color(KOLOR_RAMKA)
    ax.tick_params(colors=KOLOR_TEKST_2, labelsize=10, length=0)
    ax.xaxis.grid(True, color=KOLOR_RAMKA, linewidth=0.8)
    ax.set_axisbelow(True)
    # Pionowa linia w zerze – granica między „pomaga” a „szkodzi”
    ax.axvline(0, color=KOLOR_TEKST_2, linewidth=1)


def wykres_slupkowy(wartosci: pd.Series, podpis_osi: str, format_liczb: str):
    """
    Rysuje poziomy wykres słupkowy:
    zielone słupki = wpływ dodatni (w stronę prezentu),
    bordowe = wpływ ujemny (w stronę rózgi).
    """
    # Font wykresu: DM Sans, jeśli jest zainstalowany w systemie (inaczej domyślny bezszeryfowy)
    plt.rcParams["font.family"] = FONT_WYKRESOW

    wartosci = wartosci.sort_values()
    kolory = [KOLOR_ZIELONY if v >= 0 else KOLOR_BORDO for v in wartosci]

    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=150)
    przygotuj_os(fig, ax)
    slupki = ax.barh(wartosci.index, wartosci.values, color=kolory, height=0.6)

    # Etykiety tekstowe na końcach słupków
    zakres = max(abs(wartosci).max(), 1e-6)
    for slupek, v in zip(slupki, wartosci.values):
        przesuniecie = zakres * 0.03 * (1 if v >= 0 else -1)
        ax.text(
            v + przesuniecie,
            slupek.get_y() + slupek.get_height() / 2,
            format(v, format_liczb),
            va="center",
            ha="left" if v >= 0 else "right",
            color=KOLOR_TEKST,
            fontsize=9,
        )

    # Symetryczna oś X z zapasem na etykiety
    ax.set_xlim(-zakres * 1.35, zakres * 1.35)
    ax.set_xlabel(podpis_osi, color=KOLOR_TEKST_2, fontsize=9)
    ax.tick_params(axis="y", colors=KOLOR_TEKST, labelsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. INTERFEJS UŻYTKOWNIKA
# ---------------------------------------------------------------------------

# Wartości domyślne kontrolek (używane też przez przycisk „Reset”)
DOMYSLNE = {
    "godziny_gier": 12,
    "odpisane_zadania": 3,
    "porcje_warzyw": 10,
    "sprzatanie": "Nie",
    "kolka_naukowe": "Nie",
}


def resetuj() -> None:
    """Przywraca domyślne wartości wszystkich kontrolek (dla kolejnego gościa)."""
    for klucz, wartosc in DOMYSLNE.items():
        st.session_state[klucz] = wartosc


def main() -> None:
    wstrzyknij_css()

    # Dane i model – dzięki cache liczone tylko raz, przy pierwszym uruchomieniu
    dane = wygeneruj_dane()
    model, dokladnosc = wytrenuj_model(dane)

    # Inicjalizacja stanu kontrolek przy pierwszym wejściu
    for klucz, wartosc in DOMYSLNE.items():
        st.session_state.setdefault(klucz, wartosc)

    # --- Nagłówek -----------------------------------------------------------
    st.markdown(
        '<div class="naglowek"><div>'
        '<p class="nadtytul">Edycja świąteczna · ML, czyli machine learning</p>'
        f'<div class="tytul">MLikołaj {IKONA_GWIAZDA}</div>'
        '<p class="podtytul">Model uczenia maszynowego przeanalizował zachowanie '
        f"{LICZBA_REKORDOW} uczniów. Ustaw swoje parametry i sprawdź, czy według "
        "algorytmu czeka Cię prezent, czy rózga.</p>"
        f"</div>{logo_html('logo-naglowek')}</div>",
        unsafe_allow_html=True,
    )
    ornament()

    # --- Układ dwóch kolumn: wejście (lewa) | wynik (prawa) -----------------
    kol_lewa, kol_prawa = st.columns([1, 1.15], gap="large")

    with kol_lewa:
        st.subheader("Twoje zachowanie")
        with st.container(border=True):
            st.slider(
                CECHY["godziny_gier"], 0, 40, key="godziny_gier",
                help="Ile godzin tygodniowo spędzasz na grach komputerowych?",
            )
            st.slider(
                CECHY["odpisane_zadania"], 0, 20, key="odpisane_zadania",
                help="Ile zadań domowych w tym semestrze odpisałeś/-aś na przerwie?",
            )
            st.slider(
                CECHY["porcje_warzyw"], 0, 30, key="porcje_warzyw",
                help="Ile porcji warzyw zjadasz w ciągu tygodnia?",
            )
            st.radio(
                "Sprzątasz pokój bez przypominania?",
                ["Tak", "Nie"], key="sprzatanie", horizontal=True,
            )
            st.radio(
                "Chodzisz na kółko matematyczne / informatyczne?",
                ["Tak", "Nie"], key="kolka_naukowe", horizontal=True,
            )
        st.button("Reset (następna osoba)", on_click=resetuj)

    # Zebranie danych wejściowych w jeden wiersz – w tej samej kolejności co przy treningu
    uczen = pd.DataFrame(
        [
            {
                "godziny_gier": st.session_state["godziny_gier"],
                "odpisane_zadania": st.session_state["odpisane_zadania"],
                "porcje_warzyw": st.session_state["porcje_warzyw"],
                "sprzatanie": 1 if st.session_state["sprzatanie"] == "Tak" else 0,
                "kolka_naukowe": 1 if st.session_state["kolka_naukowe"] == "Tak" else 0,
            }
        ]
    )[list(CECHY)]

    # Predykcja: prawdopodobieństwo klasy 1 (prezent)
    p_prezent = float(model.predict_proba(uczen)[0, 1])
    procent = round(p_prezent * 100)

    # Werdykt i kolor zależny od wyniku
    if p_prezent >= 0.65:
        werdykt, kolor, ikona = "Prezent prawie pewny", KOLOR_ZIELONY, IKONA_PREZENT
    elif p_prezent >= 0.40:
        werdykt, kolor, ikona = "Na granicy – wszystko się może zdarzyć", KOLOR_ZLOTY, IKONA_GWIAZDA
    else:
        werdykt, kolor, ikona = "Uwaga: rośnie ryzyko rózgi", KOLOR_BORDO, IKONA_ROZGA

    # --- Wkład poszczególnych cech dla TEGO ucznia --------------------------
    # W regresji logistycznej: logit = wyraz_wolny + suma(waga_i * cecha_i_przeskalowana).
    # Każdy składnik tej sumy to „wkład” danej cechy w ostateczną decyzję.
    skaler = model.named_steps["skaler"]
    regresja = model.named_steps["regresja"]
    wagi = pd.Series(regresja.coef_[0], index=[CECHY[c] for c in CECHY])
    uczen_przeskalowany = skaler.transform(uczen)[0]
    wklad = pd.Series(regresja.coef_[0] * uczen_przeskalowany, index=wagi.index)

    with kol_prawa:
        st.subheader("Decyzja modelu")
        st.markdown(
            f"""
            <div class="karta-wyniku">
                <div class="ikona" style="color:{kolor}">{ikona}</div>
                <div class="etykieta">Prawdopodobieństwo prezentu</div>
                <div class="liczba" style="color:{kolor}">{procent}%</div>
                <div class="werdykt" style="color:{kolor}">{werdykt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Pasek postępu aktualizuje się przy każdej zmianie suwaka
        pokoloruj_pasek_postepu(kolor)
        st.progress(p_prezent, text=f"Prezent {procent}% · Rózga {100 - procent}%")

        st.markdown("##### Co wpłynęło na Twój wynik?")
        st.caption(
            "Wkład każdej cechy w porównaniu z przeciętnym uczniem. "
            "Zielone przesuwają decyzję w stronę prezentu, bordowe – w stronę rózgi."
        )
        fig = wykres_slupkowy(wklad, "wkład w decyzję (logit)", "+.2f")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # --- Sekcja edukacyjna: globalne wagi modelu -----------------------------
    ornament()
    kol_wykres, kol_opis = st.columns([1.15, 1], gap="large")

    with kol_wykres:
        st.subheader("Jak myśli algorytm? Wagi modelu")
        fig = wykres_slupkowy(wagi, "współczynnik regresji (cechy standaryzowane)", "+.2f")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with kol_opis:
        st.subheader("Pod maską")
        st.markdown(
            f"""
            - **Model:** regresja logistyczna (`scikit-learn`)
            - **Dane:** {LICZBA_REKORDOW} syntetycznych uczniów (80% nauka / 20% test)
            - **Dokładność na danych testowych:** {dokladnosc:.0%}

            Model liczy ważoną sumę cech – każda cecha ma swoją **wagę**.
            Dodatnia waga zwiększa szansę na prezent, ujemna ją zmniejsza.
            Im dłuższy słupek, tym ważniejsza cecha.

            Na koniec suma trafia do **funkcji sigmoidalnej**, która zamienia
            dowolną liczbę na prawdopodobieństwo od 0% do 100%:
            """
        )
        st.latex(r"P(\text{prezent}) = \frac{1}{1 + e^{-(w_0 + w_1x_1 + \dots + w_5x_5)}}")
        st.caption(
            "Ciekawostka: model nigdy nie widział reguły, według której wygenerowaliśmy dane. "
            "Wagi odkrył sam – wyłącznie na podstawie przykładów."
        )

    # Opcjonalny podgląd danych treningowych dla najbardziej dociekliwych
    with st.expander("Podejrzyj dane treningowe"):
        podglad = dane.rename(columns={**CECHY, "prezent": "Prezent (1) / Rózga (0)"})
        st.dataframe(podglad.head(50), width="stretch", hide_index=True)

    # Stopka z życzeniami
    ornament()
    st.markdown(
        '<p class="stopka">Wesołych Świąt i samych trafnych predykcji w Nowym Roku!</p>'
        f'<div class="autor">{logo_html("logo-stopka")}<span>Autor: {AUTOR}</span></div>',
        unsafe_allow_html=True,
    )


# Punkt wejścia skryptu
if __name__ == "__main__":
    main()
