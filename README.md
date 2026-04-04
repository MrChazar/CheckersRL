# Wykaz zmian

### Polityka DQN 
Model nie uczy się w zależności od strony biały/czarny. Uczy się wyłącznie białej strony, a jeśli gra jako czarny to po prostu wybiera nie najlepszy a najgorszy możliwy ruch (z perspektywy białej) informacja o stronie wewnątrz vec_state jest wyzerowana przez cały trening, natomiast w DQNAgent dodałem atrybut side mówiący o stronie agenta [nie pamiętam czy zmiana taka miała istotny wpływ na efekty uczenia]

### Replay buffer
Zmodyfikowany na PRIORITIZED replay buffer czyli tak że jest większe prawdopodobieństwo ponownego wyboru tych próbek, które zwróciły większy błąd [nie miało to zauważalnego wpływu na loss ani win rate]. Po poprawieniu usuwania starych próbek na najgorsze próbki pod względem priorytetu nie widać już takich skoków lossu, gdy buffer się przepełnia.
 Dodana zmiana parametru beta wraz z epokami. Beta wpływa na korekcję priorytetyzowania próbek, im większa epoka tym większa korekcja aż osiągnie pełne *importance sampling*.

### Target network
Modyfikacja kopiowania modelu co ‚n’ epok na ustawianie wag sieci target co epokę w następujący sposób: (1 - tau) wagi target z poprzedniej epoki + (tau) wagi policy net [ustawienie parametru tau na 0.5 lub 0.6 poprawia loss! - loss spada przez cały okres treningu aczkolwiek im dalej w trening tym spadek ten był mniejszy, natomiast ustawienie na 0.75 to było już za dużo i wtedy loss odbijał znów w górę]

### kroki uczenia
Zmodyfikowałem tworzenie próbek (transitions) na multi step learning. stan końcowy opisuje planszę nie po 1 a 4 ruchach w przód (2 własne + 2 przeciwnika) a nagrodą jest suma nagród wszystkich 4 kroków [jak dobrze pamiętam to też pomogło]

### train_step()
Dodałem double DQN [poprawiło trochę wyniki]

### DQNNet
Dodałem dueling DQN [poprawiło trochę wyniki]

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