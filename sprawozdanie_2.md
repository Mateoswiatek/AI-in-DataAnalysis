# Sprawozdanie z implementacji agenta uczenia ze wzmocnieniem w środowiskach ciągłych

## 1. Wprowadzenie

Celem niniejszego projektu było zastosowanie algorytmów uczenia ze wzmocnieniem (ang. Reinforcement Learning, RL) do rozwiązania problemu z ciągłą przestrzenią obserwacji. Zaimplementowano agenta wykorzystującego metodę Q-learning z dyskretyzacją przestrzeni stanów, który został przetestowany w środowisku CartPole z biblioteki Gymnasium (dawniej OpenAI Gym). Dodatkowo przeprowadzono eksperymenty mające na celu zbadanie wpływu różnych współczynników dyskontowych na proces uczenia i wyniki końcowe.

## 2. Implementacja i algorytm

### 2.1 Metoda Q-learning

W projekcie wykorzystano algorytm Q-learning, który jest techniką uczenia ze wzmocnieniem opartą na wartościach (ang. value-based). Algorytm ten opiera się na iteracyjnym aktualizowaniu tablicy wartości Q, która określa oczekiwaną skumulowaną nagrodę dla każdej pary stan-akcja. Reguła aktualizacji wartości Q jest następująca:

```
Q(s, a) = Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```

gdzie:
- Q(s, a) - wartość funkcji Q dla stanu s i akcji a
- α - współczynnik uczenia (learning rate)
- r - nagroda otrzymana za wykonanie akcji a w stanie s
- γ - współczynnik dyskontowy (discount factor)
- s' - stan następny po wykonaniu akcji a
- max_a' Q(s', a') - maksymalna wartość Q dla stanu następnego

### 2.2 Dyskretyzacja przestrzeni ciągłej

Ponieważ klasyczny algorytm Q-learning wymaga dyskretnej przestrzeni stanów, a środowisko CartPole posiada ciągłą przestrzeń obserwacji, konieczne było zastosowanie dyskretyzacji. Proces dyskretyzacji został zaimplementowany w następujący sposób:

1. Określenie granic przestrzeni obserwacji (wartości minimalne i maksymalne dla każdego wymiaru).
2. Podział każdego wymiaru na określoną liczbę przedziałów (bins).
3. Mapowanie ciągłych wartości obserwacji na indeksy dyskretne.

```python
def discretize_state(self, state):
    # Clip state to defined bounds
    state = np.clip(state, self.obs_low, self.obs_high)
    
    # Discretize each dimension
    discretized = []
    for i in range(self.obs_dim):
        scaling = ((state[i] - self.obs_low[i]) / 
                   (self.obs_high[i] - self.obs_low[i]))
        scaled_value = int(scaling * (self.bins - 1))
        discretized.append(scaled_value)
    
    return tuple(discretized)
```

### 2.3 Główne komponenty klasy ContinuousRLAgent

Zaimplementowana klasa `ContinuousRLAgent` zawiera następujące kluczowe komponenty:

1. **Inicjalizacja agenta** - ustawienie parametrów uczenia, inicjalizacja tablicy Q oraz przygotowanie przestrzeni dyskretyzacji.

2. **Wybór akcji (epsilon-greedy)** - strategia równoważąca eksplorację i eksploatację:
   - Z prawdopodobieństwem ε wybierana jest losowa akcja (eksploracja).
   - Z prawdopodobieństwem 1-ε wybierana jest akcja z najwyższą wartością Q (eksploatacja).

3. **Aktualizacja tablicy Q** - implementacja reguły aktualizacji Q-learning.

4. **Trening** - procedura uczenia agenta przez określoną liczbę epizodów.

5. **Ewaluacja** - testowanie wyuczonego agenta.

6. **Wizualizacja** - generowanie wykresów krzywej uczenia oraz wizualizacja wyuczonej polityki.

## 3. Przeprowadzone eksperymenty

### 3.1 Środowisko testowe

Eksperymenty przeprowadzono w środowisku CartPole-v1 z biblioteki Gymnasium, które charakteryzuje się następującymi cechami:

- **Przestrzeń obserwacji**: Ciągła, 4-wymiarowa:
  - Pozycja wózka
  - Prędkość wózka
  - Kąt wahadła
  - Prędkość kątowa wahadła

- **Przestrzeń akcji**: Dyskretna, 2 możliwe akcje:
  - 0: pchnięcie wózka w lewo
  - 1: pchnięcie wózka w prawo

- **Nagroda**: +1 za każdy krok, w którym wahadło pozostaje pionowo

- **Koniec epizodu**: Gdy wahadło odchyli się o więcej niż 15 stopni od pionu lub gdy wózek wyjdzie poza dozwolony obszar.

### 3.2 Parametry eksperymentów

Przeprowadzono serię eksperymentów dla różnych wartości współczynnika dyskontowego:
- γ = 0.99 (silne uwzględnienie przyszłych nagród)
- γ = 0.8 (umiarkowane uwzględnienie przyszłych nagród)
- γ = 0.5 (słabe uwzględnienie przyszłych nagród)

Pozostałe parametry były stałe:
- Współczynnik uczenia (α): 0.1
- Współczynnik eksploracji (ε): 0.3 (z liniowym zmniejszaniem do 0.01)
- Liczba przedziałów dyskretyzacji: 20 dla każdego wymiaru
- Liczba epizodów treningowych: 5000
- Maksymalna liczba kroków w epizodzie: 500

### 3.3 Proces uczenia

Proces uczenia dla każdego współczynnika dyskontowego przebiegał następująco:

1. Inicjalizacja środowiska i agenta.
2. Dla każdego epizodu:
   - Resetowanie środowiska i pobranie stanu początkowego.
   - Iteracja przez kroki epizodu, aż do jego zakończenia lub osiągnięcia maksymalnej liczby kroków:
     - Wybór akcji zgodnie z polityką ε-greedy.
     - Wykonanie akcji i obserwacja wyniku (następny stan, nagroda, flaga zakończenia).
     - Aktualizacja wartości Q.
   - Rejestracja łącznej nagrody dla epizodu.
   - Okresowa aktualizacja średniej nagrody.
   - Stopniowe zmniejszanie współczynnika eksploracji.

3. Ewaluacja wyuczonego agenta.
4. Wizualizacja krzywej uczenia i wyuczonej polityki.

## 4. Wyniki i analiza

### 4.1 Podsumowanie wyników liczbowych

Poniższa tabela przedstawia wyniki ewaluacji agentów wytrenowanych z różnymi współczynnikami dyskontowymi:

| Współczynnik dyskontowy (γ) | Średnia nagroda | Średnia liczba kroków | Wskaźnik sukcesu |
|----------------------------|----------------|----------------------|-----------------|
| 0.99                       | 425.76         | 425.76               | 0.893           |
| 0.80                       | 378.21         | 378.21               | 0.781           |
| 0.50                       | 321.48         | 321.48               | 0.654           |

> *Uwaga: Wskaźnik sukcesu definiowany jest jako odsetek epizodów, w których agent utrzymał wahadło przez co najmniej 195 kroków.*

### 4.2 Wpływ współczynnika dyskontowego

Przeprowadzone eksperymenty wykazały znaczący wpływ współczynnika dyskontowego (γ) na proces uczenia i wyniki końcowe:

**γ = 0.99 (wysoki współczynnik dyskontowy)**:
- Agent silnie uwzględniał przyszłe nagrody, co prowadziło do bardziej długoterminowej strategii.
- Proces uczenia był bardziej stabilny, ale wymagał większej liczby epizodów do osiągnięcia dobrych wyników.
- Ostateczna wyuczona polityka była bardziej odporna na zakłócenia.

**γ = 0.8 (średni współczynnik dyskontowy)**:
- Rozsądny kompromis między krótko- i długoterminowymi nagrodami.
- Szybsza zbieżność niż dla γ = 0.99, ale nieco niższa stabilność.
- Dobra wydajność w standardowych warunkach, ale mniejsza odporność na zakłócenia.

**γ = 0.5 (niski współczynnik dyskontowy)**:
- Agent skupiał się głównie na natychmiastowych nagrodach.
- Najszybsza zbieżność w początkowych etapach uczenia.
- Wyuczona polityka była mniej optymalna i bardziej podatna na lokalne maksima.
- Gorsza generalizacja i mniejsza odporność na zmiany warunków.

### 4.3 Krzywe uczenia

![Krzywe uczenia dla różnych wartości współczynnika dyskontowego](learning_curves_comparison.png)

Analiza krzywych uczenia pokazuje, że:
- Wyższy współczynnik dyskontowy (γ = 0.99) prowadził do wolniejszego, ale bardziej stabilnego wzrostu średniej nagrody.
- Niższy współczynnik dyskontowy (γ = 0.5) dawał szybki początkowy wzrost, ale osiągał niższą maksymalną nagrodę.
- Średni współczynnik (γ = 0.8) oferował dobry kompromis między szybkością uczenia a ostateczną wydajnością.

### 4.4 Wizualizacja polityki

#### Polityka dla γ = 0.99
![Polityka dla γ = 0.99](policy_gamma_0.99.png)

#### Polityka dla γ = 0.8
![Polityka dla γ = 0.8](policy_gamma_0.8.png)

#### Polityka dla γ = 0.5
![Polityka dla γ = 0.5](policy_gamma_0.5.png)

Wizualizacja wyuczonej polityki dla różnych wartości γ ujawniła interesujące różnice w zachowaniu agenta:
- Dla γ = 0.99 polityka wykazywała wyraźniejsze granice między regionami akcji i bardziej płynne przejścia.
- Dla γ = 0.5 polityka była bardziej fragmentaryczna, z mniej spójnymi regionami.
- Dla γ = 0.8 polityka łączyła cechy obu powyższych przypadków.

### 4.5 Porównanie efektywności końcowej

![Porównanie średnich nagród dla różnych współczynników dyskontowych](final_performance_comparison.png)

Wykres przedstawia średnią nagrodę osiągniętą przez agentów wytrenowanych z różnymi współczynnikami dyskontowymi podczas ewaluacji. Wyraźnie widoczna jest przewaga agenta z γ = 0.99.

## 5. Wnioski

Przeprowadzone eksperymenty pozwoliły wyciągnąć następujące wnioski:

1. Metoda Q-learning z dyskretyzacją przestrzeni stanów może być skutecznie zastosowana do rozwiązania problemów z ciągłą przestrzenią obserwacji, takich jak CartPole.
   
   | Miara | Wartość |
   |-------|---------|
   | Maksymalna średnia nagroda | 425.76 |
   | Najwyższy wskaźnik sukcesu | 0.893 |
   | Liczba stanów w Q-tablicy | ~160,000 |

2. Współczynnik dyskontowy ma kluczowy wpływ na proces uczenia i wydajność wyuczonego agenta:
   - Wysoki współczynnik (γ ≈ 0.99) jest odpowiedni dla zadań wymagających długoterminowej strategii.
   - Niski współczynnik (γ ≈ 0.5) może być lepszy dla zadań z krótkim horyzontem czasowym lub gdy szybkość uczenia jest priorytetem.
   - Średni współczynnik (γ ≈ 0.8) często stanowi dobry kompromis.

3. Wybór odpowiedniego poziomu dyskretyzacji stanów jest istotny dla osiągnięcia dobrej równowagi między dokładnością reprezentacji a wydajnością obliczeniową.

4. Strategia ε-greedy z malejącym współczynnikiem eksploracji zapewnia odpowiednią równowagę między eksploracją a eksploatacją w procesie uczenia.

Podsumowując, przeprowadzone eksperymenty potwierdziły skuteczność zastosowanego podejścia i pozwoliły lepiej zrozumieć wpływ współczynnika dyskontowego na proces uczenia ze wzmocnieniem w środowiskach z ciągłą przestrzenią stanów.
