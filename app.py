"""
Algorytm Grzeczności
====================

Interaktywna aplikacja edukacyjna (Streamlit + scikit-learn) na stoisko wydziałowe.

Co się tu dzieje?
1. Generujemy syntetyczny (zmyślony, ale logiczny) zbiór danych o 1000 uczniach.
2. Trenujemy na nim regresję logistyczną – prosty i przejrzysty model klasyfikacyjny.
3. Użytkownik ustawia swoje „zachowanie” suwakami i przyciskami,
   a model na bieżąco liczy prawdopodobieństwo otrzymania prezentu (klasa 1)
   zamiast rózgi (klasa 0).
4. Wykresy pokazują, JAK model podjął decyzję – czyli wagi poszczególnych cech.

Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py
"""

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
    page_title="Algorytm Grzeczności",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paleta kolorów – trzymamy ją w jednym miejscu, żeby UI i wykresy były spójne.
KOLOR_TLO = "#0E1117"        # tło całej aplikacji
KOLOR_PANEL = "#161B22"      # tło „kart” (paneli)
KOLOR_RAMKA = "#262C36"      # delikatne obramowania i siatka wykresów
KOLOR_TEKST = "#E6EDF3"      # główny tekst
KOLOR_TEKST_2 = "#8B949E"    # tekst pomocniczy (opisy, podpisy osi)
KOLOR_ZIELONY = "#2EA043"    # „prezent” / wpływ dodatni
KOLOR_POMARANCZ = "#F0883E"  # „rózga” / wpływ ujemny
KOLOR_NIEBIESKI = "#58A6FF"  # akcent interfejsu (suwaki, przyciski)
KOLOR_ZLOTY = "#E3B341"      # wynik „na granicy”

# Ziarno losowości – dzięki niemu dane i model są za każdym razem identyczne.
ZIARNO = 42
LICZBA_REKORDOW = 1000

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

        /* Ciemne tło aplikacji */
        .stApp {{
            background-color: {KOLOR_TLO};
            color: {KOLOR_TEKST};
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
            color: {KOLOR_NIEBIESKI};
            font-size: 0.8rem;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: -0.4rem;
        }}

        .podtytul {{
            color: {KOLOR_TEKST_2};
            font-size: 1.05rem;
            max-width: 52rem;
        }}

        /* Karta z dużym wynikiem procentowym */
        .karta-wyniku {{
            background: {KOLOR_PANEL};
            border: 1px solid {KOLOR_RAMKA};
            border-radius: 14px;
            padding: 1.4rem 1.6rem 1.1rem 1.6rem;
            margin-bottom: 0.8rem;
        }}
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

        /* Suwaki – niebieski akcent */
        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {KOLOR_NIEBIESKI} !important;
            border-color: {KOLOR_NIEBIESKI} !important;
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
            color: {KOLOR_NIEBIESKI};
            border: 1px solid {KOLOR_NIEBIESKI};
            border-radius: 8px;
            font-weight: 500;
        }}
        .stButton > button:hover {{
            background: {KOLOR_NIEBIESKI};
            color: {KOLOR_TLO};
            border-color: {KOLOR_NIEBIESKI};
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

        /* Cienka linia oddzielająca sekcje */
        hr {{ border-color: {KOLOR_RAMKA} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    pomarańczowe = wpływ ujemny (w stronę rózgi).
    """
    # Font wykresu: DM Sans, jeśli jest zainstalowany w systemie (inaczej domyślny bezszeryfowy)
    plt.rcParams["font.family"] = FONT_WYKRESOW

    wartosci = wartosci.sort_values()
    kolory = [KOLOR_ZIELONY if v >= 0 else KOLOR_POMARANCZ for v in wartosci]

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
    st.markdown('<p class="nadtytul">Uczenie maszynowe · demo</p>', unsafe_allow_html=True)
    st.title("Algorytm Grzeczności")
    st.markdown(
        '<p class="podtytul">Model uczenia maszynowego przeanalizował zachowanie '
        f"{LICZBA_REKORDOW} uczniów. Ustaw swoje parametry i sprawdź, czy według "
        "algorytmu czeka Cię prezent, czy rózga.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

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
        werdykt, kolor = "Prezent prawie pewny", KOLOR_ZIELONY
    elif p_prezent >= 0.40:
        werdykt, kolor = "Na granicy – wszystko się może zdarzyć", KOLOR_ZLOTY
    else:
        werdykt, kolor = "Uwaga: rośnie ryzyko rózgi", KOLOR_POMARANCZ

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
            "Zielone przesuwają decyzję w stronę prezentu, pomarańczowe – w stronę rózgi."
        )
        fig = wykres_slupkowy(wklad, "wkład w decyzję (logit)", "+.2f")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # --- Sekcja edukacyjna: globalne wagi modelu -----------------------------
    st.divider()
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


# Punkt wejścia skryptu
if __name__ == "__main__":
    main()
