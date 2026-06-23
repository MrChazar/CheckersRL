# Opis
## Reprezentacja stanu (z biblioteki deepdraughts)
1. vec_board  — reprezentacja pozycji pionów na planszy
   - Tensor (4, nsize, nsize)
   - kanał 0 — biały zwykły pion
   - kanał 1 — biała damka
   - kanał 2 — czarny zwykły pion
   - kanał 3 — czarna damka
2. vec_state  — dodatkowy wektor stanu gry
   - długości 3 + 2 * nsize
   - player -1 / 1 - który gracz wykonuje ruch
   - is_chain_taking 0/1 - czy zbijanie kilku pod rząd
   - taken_piece nsize * 2 - pozycje zbitych figur w łańcuchu bicia
   - n_king_move 0/1 - czy ruch damką

## Nauka sieci - transitions
Transition składa się z:
- state_vec_board, state_vec_extra - reprezentacja stanu
- action - ruch, który został podjęty
- reward - nagroda, ważona dla tego ruchu
- next_state_vec_board, next_state_vec_extra - stan gry po kolejnych n ruchach
- done - czy gra zakończona
- b_next_legal_mask - maska legalnych ruchów dla stanu next
Sieć liczy
1. `q_values = self.policy_net(state_board, state_extra)` ile obecna sieć uważa, że warta była akcja wykonana w danym stanie.
2. wartość obecnej akcji = nagroda teraz + zdyskontowana wartość najlepszej przyszłej akcji. W Dueling DQN:
    - `next_q_online = self.policy_net(next_board, next_extra)
        next_actions = next_q_online.argmax(1, keepdim=True)` policy_net wybiera która akcja wygląda najlepiej
    - `next_q_target = self.target_net(next_board, next_extra)
       next_q_value = next_q_target.gather(1, next_actions)` target_net ocenia ile ta wybrana akcja jest warta. To ogranicza zawyżanie Q, target_net aktualizuje się przez jakiś ułamek policy_net.
3. `expected_q_val = reward + (gamma ** n_steps) * next_q_value * (1.0 - done)` liczy oczekiwane Q 
4. `td_errors = q_val - expected_q_val` TD Błąd: porównanie nagrody + najlepsze Q potem vs najlepsze Q przewidziane teraz
5. Dodatkowo ważenie błędu według wagi próbki

## Nagrody
GAME_WIN = 1.
GAME_TIE = 0.
GAME_LOSS = -1.
PIECE_TAKEN = .01

# Wykaz zmian

### Replay buffer - ważony
Zmodyfikowany na PRIORITIZED replay buffer czyli tak że jest większe prawdopodobieństwo ponownego wyboru tych próbek, które zwróciły większy błąd [nie miało to zauważalnego wpływu na loss ani win rate]. Po poprawieniu usuwania starych próbek na najgorsze próbki pod względem priorytetu nie widać już takich skoków lossu, gdy buffer się przepełnia.
 Dodana zmiana parametru beta wraz z epokami. Beta wpływa na korekcję priorytetyzowania próbek, im większa epoka tym większa korekcja aż osiągnie pełne *importance sampling*.
- Zaleta: uczenie jest szybsze, bo sygnał nagrody szybciej wraca do wcześniejszych ruchów
- Wada: target ma większą wariancję i zależy od kolejnych rzeczywistych decyzji agenta

### Target network wydzielony od policy network
Modyfikacja kopiowania modelu co ‚n’ epok na ustawianie wag sieci target co epokę w następujący sposób: (1 - tau) wagi target z poprzedniej epoki + (tau) wagi policy net [ustawienie parametru tau na 0.5 lub 0.6 poprawia loss! - loss spada przez cały okres treningu aczkolwiek im dalej w trening tym spadek ten był mniejszy, natomiast ustawienie na 0.75 to było już za dużo i wtedy loss odbijał znów w górę]

## Kroki uczenia - n ruchów
Zmodyfikowałem tworzenie próbek (transitions) na multi step learning. stan końcowy opisuje planszę nie po 1 a 4 ruchach w przód (2 własne + 2 przeciwnika) a nagrodą jest suma nagród wszystkich 4 kroków [jak dobrze pamiętam to też pomogło]

### train_step()
Dodałem double DQN

### DQNNet
Dodałem dueling DQN

### W parametrach
Zmiana z use_gpu na device, bo lokalnie nie mam cuda, ale za to mps, więc żeby można było sobie samemu zdefiniować swój device

### Kod treningowy
Dodałem funkcję do ewaluacji modelu (ewaluacja tym mocnym modelem MCTS aczkolwiek z pomniejszoną liczbą n_playout aby go troszkę osłabić)

### Ewaluacja
Ewaluacja uruchamia się co określoną liczbę epok przeciwko MCTS z ustaloną głębokością, mierzy *win ratio* i *average Q*. Plik run_test_eval.py uruchamia samą ewaluację.

### Checkpointy
Ulepszyłem zapisywanie checkpointów, aby dało się wznowić trening, a ten potrafi trwać bardzo długo.

# Jak używać
### Uruchamianie:
- Granie z botem na GUI - `python run.py`
- Uruchomienie treningu - `python run_training.py`
- Uruchomienie ewaluacji - `python run_test_eval.py`
### Pliki:
- config.ini - plik konfiguracyjny pozwalający wybrać model dla treningu oraz zwykłej gry
- run.py - pozwala nam zagrać z gui z wytrenowanym botem
- run_training.py -  pozwala na skonfigurować uczenie naszego modelu, więcej szczegółów jest w train_pipeline.py
- run_test_eval.py -  uruchamia ewaluację testową przeciwko MCTS
- train_pipeline.py - definiuje wszystkie parametry uczenia o których powinniśmy wiedzieć
- game_collector.py - odpowiada za odpalanie gierek, pobieranie stanów
- mcts_alphazero i net_pytorch - implementacja uczenia w stylu alphazero (fajne)
- generator.py - plik pozwalający na odpalanie wielu rozgrywek na raz

# Todo
- Lepszy bot xD. Pomysły:
  - Zrobić lepsze zapisywanie stanu, aby bez problemu kontynuować uczenie.
  - Trenować w 2 fazach, najpierw z szybkim zejściem epsilona aż zacznie coś grać, potem fine tuning z małym LR i tau
- Super byłaby jakaś wizualizacja tego jak myśli model podczas ruchu. + eval bar
- Można rozbudować GUI o wybór bota z którym gramy (chociażby MCTS/DQN).
- Najlepiej jakby się dało puścić przeciwko sobie boty na GUI, z opcją przewijania ruchów.