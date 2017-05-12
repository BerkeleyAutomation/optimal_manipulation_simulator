import matplotlib.pyplot as plt
sCosts = []
with open('.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split('\n'))