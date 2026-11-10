# Algorytm Grzeczności – opis na plakat

## Hasło i krótki opis

**Algorytm Grzeczności** to interaktywna aplikacja, w której model uczenia maszynowego ocenia, czy w tym roku czeka Cię prezent, czy rózga. Ustawiasz suwakami swoje zachowanie, a algorytm w ułamku sekundy podaje szansę na prezent w procentach i pokazuje, co zdecydowało o wyniku.

Propozycje hasła na plakat:

- Prezent czy rózga? Zapytaj algorytm.
- Sprawdź, co myśli o Tobie sztuczna inteligencja.
- Zobacz, jak maszyna uczy się oceniać.

Linijka pod tytułem: *Interaktywne demo uczenia maszynowego · Rada Samorządu Studenckiego WMFI UG*

## Jak to działa

Algorytm uczy się na przykładach, a potem ocenia Ciebie tak samo, jak oceniłby każdego innego ucznia.

**Dane → Nauka → Ty → Wynik → Dlaczego?**

1. **Dane:** komputer losuje 1000 zmyślonych uczniów. Każdy ma swoje zachowanie i informację, czy dostał prezent, czy rózgę.
2. **Nauka:** model przegląda te przykłady i sam ustala, ile znaczy każda cecha. Nikt mu nie podaje reguły.
3. **Ty:** ustawiasz suwakami i przyciskami swoje zachowanie.
4. **Wynik:** po każdej zmianie model od razu liczy szansę na prezent, od 0% do 100%.
5. **Dlaczego?** Wykres pokazuje, które cechy przybliżyły Cię do prezentu, a które do rózgi.

## Co wpływa na wynik

Na wynik wpływa pięć cech: trzy go podnoszą, dwie obniżają. Najmocniej działają odpisane zadania, które ważą ponad półtora raza więcej niż warzywa.

| Cecha | Jak ją ustawiasz | Zakres |
| --- | --- | --- |
| Granie na komputerze | suwak | 0–40 h w tygodniu |
| Odpisane zadania domowe | suwak | 0–20 |
| Porcje warzyw | suwak | 0–30 w tygodniu |
| Sprzątanie pokoju bez przypominania | Tak / Nie | – |
| Kółko matematyczne lub informatyczne | Tak / Nie | – |

Wagi cech w wytrenowanym modelu (do wykresu słupkowego na plakacie):

| Cecha | Waga | Kierunek |
| --- | --- | --- |
| Porcje warzyw w tygodniu | +1,13 | w stronę prezentu |
| Kółko mat./inf. | +0,59 | w stronę prezentu |
| Sprzątanie bez przypominania | +0,54 | w stronę prezentu |
| Granie na komputerze | −0,41 | w stronę rózgi |
| Odpisane zadania domowe | −1,85 | w stronę rózgi |

Tytuł wykresu: **Odpisane zadania najmocniej obniżają szansę na prezent.**

Długość słupka mówi, jak ważna jest cecha, a kolor, w którą stronę przesuwa decyzję: zielony w stronę prezentu, bordowy w stronę rózgi. Wagi są porównywalne, bo przed nauką wszystkie cechy sprowadzono do wspólnej skali.

## Pod maską

Algorytm to **regresja logistyczna**, jeden z najprostszych i najczęściej używanych modeli klasyfikacji. Na danych testowych trafia w **82%** przypadków.

- **Dane:** 1000 syntetycznych uczniów. Model uczy się na 800 z nich, a na pozostałych 200 sprawdzamy, czy „zdał egzamin”.
- **Model:** liczy ważoną sumę cech. Każda cecha ma swoją wagę, dodatnią albo ujemną.
- **Sigmoida:** zamienia tę sumę na prawdopodobieństwo od 0% do 100%.
- **Narzędzia:** Python, scikit-learn i Streamlit.

Wzór na plakat:

```latex
P(\text{prezent}) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + \dots + w_5 x_5)}}
```

x to cechy ucznia (np. liczba porcji warzyw), w to wagi, których model nauczył się sam.

## Czego się nauczysz

Najważniejsza lekcja: model jest tak dobry i tak stronniczy, jak dane, na których się uczył.

- **Maszyna uczy się z przykładów.** Nikt nie wpisał jej reguły „warzywa są dobre”. Odkryła ją sama z danych.
- **Decyzję da się wyjaśnić.** Wykres wag pokazuje, dlaczego algorytm zdecydował tak, a nie inaczej.
- **Wynik to prawdopodobieństwo, nie pewność.** Nawet 95% oznacza, że czasem zdarzy się rózga.

Pytania, które można zadać na plakacie lub przy stoisku:

- Skąd algorytm wie, że warzywa są dobre?
- Co by się stało, gdyby w danych wszyscy gracze dostawali prezenty?
- Gdzie w prawdziwym życiu algorytmy oceniają ludzi? (kredyty, rekrutacja, rekomendacje)
- Czy uda Ci się dostać 100%?

## Wskazówki do plakatu

Plakat powinien wyglądać jak aplikacja: ciemna zieleń, bordo i złoto, font DM Sans, dużo powietrza.

Proponowany układ od góry:

1. Hasło i tytuł „Algorytm Grzeczności” ze złotą gwiazdką.
2. Duży zrzut ekranu aplikacji (`docs/zrzut-ekranu.png`).
3. Schemat „Jak to działa” w pięciu krokach.
4. Wykres wag cech i jedno zdanie o odpisanych zadaniach.
5. Wzór z sigmoidą i liczba: 82% trafności.
6. Pytanie zachęcające do podejścia, np. „Czy uda Ci się dostać 100%?”
7. Stopka z logo i podpisem: Rada Samorządu Studenckiego WMFI UG.

Kolory z aplikacji:

| Rola | Kolor | HEX |
| --- | --- | --- |
| Tło | ciemny świerk | #0A1C15 |
| Panele | zieleń | #10271E |
| Tekst | kremowa biel | #F4EEE2 |
| Akcent, gwiazdki | złoto | #E2B04A |
| Prezent, wpływ dodatni | zieleń choinkowa | #5CBF80 |
| Rózga, wpływ ujemny | bordo | #C2405A |
| Wstążki | ciemne bordo | #7A1F33 |

Font: [DM Sans](https://fonts.google.com/specimen/DM+Sans) (Google Fonts, bezpłatny). Logo: `assets/logo_rss.png` (białe, na przezroczystym tle). Ozdobniki: złote gwiazdki, delikatne płatki śniegu, wstążka przy wyniku. Oszczędnie, żeby plakat został czytelny.

---

Autor: Rada Samorządu Studenckiego WMFI UG
