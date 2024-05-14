import matplotlib.pyplot as plt

filename = "metrics/psnr.txt"

floats = []

with open(filename, 'r') as file:
     for line in file:
        val = float(line.strip())
        floats.append(val)

avg = []
temp = 0.0

for i in range(len(floats)):
    if i != 0 and i % 100 == 0:
        temp /= 100
        avg.append(temp)
        temp = 0.0
    temp += floats[i]


temp /= 100
avg.append(temp)
temp = 0.0

plt.figure(figsize=(10, 7))
plt.plot(avg, label='PSNR', color='red')
plt.xlabel('EPOCHS')
plt.ylabel('Mean Square Error')
plt.title('Avg Mean Square Error over 250 epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'psnr.png')
plt.show()