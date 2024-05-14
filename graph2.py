import re
import matplotlib.pyplot as plt

filename = "d_loss_data.txt"

floats = []

with open(filename, 'r') as file:
    for line in file:
        match = re.search(r"[-+]?\d*\.\d+", line)  # Regular expression to find floats in each line
        if match:
            val = float(match.group())
            floats.append(val)

avg = []
temp = 0.0

for i in range(len(floats)):
    if i != 0 and i % 61 == 0:
        temp /= 100
        avg.append(temp)
        temp = 0.0
    temp += floats[i]

temp /= 100
avg.append(temp)

plt.figure(figsize=(10, 7))
plt.plot(avg, label='D LOSS', color='red')
plt.xlabel('EPOCHS')
plt.ylabel('Mean Square Error')
plt.title('Avg Mean Square Error over 250 epochs')
plt.legend()
plt.grid(True)
plt.savefig('dloss.png')
plt.show()