# wykaz zmian


## polityka DQN 
model nie uczy się w zależności od strony biały/czarny. Uczy się wyłącznie białej strony, a jeśli gra jako czarny to po prostu wybiera nie najlepszy a najgorszy możliwy ruch (z perspektywy białej) informacja o stronie wewnątrz vec_state jest wyzerowana przez cały trening, natomiast w DQNAgent dodałem atrybut side mówiący o stronie agenta [nie pamiętam czy zmiana taka miała istotny wpływ na efekty uczenia]

## replay buffer
zmodyfikowany na PRIORITIZED replay buffer czyli tak że jest większe prawdopodobieństwo ponownego wyboru tych próbek, które zwróciły większy błąd [nie miało to zauważalnego wpływu na loss ani win rate]

## target network
modyfikacja kopiowania modelu co ‚n’ epok na ustawianie wag sieci target co epokę w następujący sposób: (1 - tau) wagi target z poprzedniej epoki + (tau) wagi policy net [ustawienie parametru tau na 0.5 lub 0.6 poprawia loss! - loss spada przez cały okres treningu aczkolwiek im dalej w trening tym spadek ten był mniejszy, natomiast ustawienie na 0.75 to było już za dużo i wtedy loss odbijał znów w górę]

## kroki uczenia
zmodyfikowałem tworzenie próbek (transitions) na multi step learning. stan końcowy opisuje planszę nie po 1 a 4 ruchach w przód (2 własne + 2 przeciwnika) a nagrodą jest suma nagród wszystkich 4 kroków [jak dobrze pamiętam to też pomogło]

## train_step()
dodałem double DQN [poprawiło trochę wyniki]

## DQNNet
dodałem dueling DQN [poprawiło trochę wyniki]

## w parametrach
zmiana z use_gpu na device, bo lokalnie nie mam cuda, ale za to mps, więc żeby można było sobie samemu zdefiniować swój device

## kod treningowy
dodałem funkcję do ewaluacji modelu (ewaluacja tym mocnym modelem MCTS aczkolwiek z pomniejszoną liczbą n_playout aby go troszkę osłabić)
