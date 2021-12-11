import pandas as pd

df = pd.read_csv('./search/result.csv')

minimax = df[df.Agent == 'ExpectimaxAgent'][['Time', 'Score']]
expectimax = df[df.Agent == 'MiniMaxAgent'][['Time', 'Score']]

print(expectimax)