# Klasyfikator Cyfr MNIST

## Opis projektu
Ten projekt to prosty klasyfikator obrazów ręcznie pisanych cyfr z wykorzystaniem zbioru danych MNIST. Zaimplementowany jest w Pythonie z użyciem PyTorch i demonstruje proces trenowania, ewaluacji oraz zapisywania modelu sieci neuronowej do rozpoznawania cyfr.

## Jak uruchomić projekt

### 1. Sklonuj repozytorium
```
git clone https://github.com/krystiandsw/Projekt-Programistyczny-2
cd Projekt-Programistyczny-2
```

### 2. Zainstaluj zależności
Upewnij się, że masz zainstalowanego Pythona 3.8+ oraz pip. Następnie zainstaluj wymagane pakiety:
```
pip install torch torchvision
```

### 3. Uruchom trenowanie i ewaluację
Możesz uruchomić główny skrypt, aby wytrenować model i ocenić jego skuteczność:
```
python main.py
```
To spowoduje:
- Pobranie zbioru danych MNIST (jeśli nie jest jeszcze obecny)
- Wytrenowanie prostej sieci neuronowej na zbiorze treningowym
- Zapisanie wytrenowanego modelu do pliku `model.pth`
- Ewaluację modelu na zbiorze testowym
- Wyświetlenie przykładowych przewidywań

## Struktura projektu
- `main.py` – Główny punkt wejścia: trenuje, zapisuje i ocenia model
- `train.py` – Zawiera pętlę treningową
- `test.py` – Zawiera funkcje ewaluacji i wyświetlania przewidywań
- `model.py` – Definiuje architekturę sieci neuronowej
- `utils.py` – Funkcje pomocnicze do zapisywania/ładowania modeli
- `data/` – Pliki zbioru danych MNIST

## Podsumowanie
Ten projekt dostarcza minimalny, ale kompletny przykład klasyfikacji obrazów z użyciem PyTorch. Jest odpowiedni dla początkujących, którzy chcą zrozumieć podstawy przepływu pracy w głębokim uczeniu, w tym ładowanie danych, trenowanie modelu, ewaluację i jego utrwalanie. Wszelkie wkłady i ulepszenia są mile widziane!
