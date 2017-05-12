import matplotlib.pyplot as plt

data = [11,1,1,1,1,1,1,1,1,1,11,1,1,1,21,1,1,1,1]
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

plt.bar(x,data)
plt.xlabel("Replan #")
plt.ylabel("# of Iterations")
plt.show()