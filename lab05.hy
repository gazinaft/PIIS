(import [pandas :as pd])
(import [numpy :as np])
(setv df (.read_csv pd "./search/result.csv"))

(setv minimax (get df (= (get df "Agent") "MiniMaxAgent")))
(setv expectimax (get df (= (get df "Agent") "ExpectimaxAgent")))

(print "Матсподівання часу для мінімаксу: " (.mean (.array np (get minimax "Time"))))
(print "Дисперсія очок для мінімаксу: " (.var np (.array np (get minimax "Score"))))


(print "Матсподівання часу для експектімаксу: " (.mean (.array np (get expectimax "Time"))))
(print "Дисперсія очок для експектімаксу: " (.var np (.array np (get expectimax "Score"))))
