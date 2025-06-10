import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
times = [-100, -100, 1, -100, -100, 339, 566, 1056]

log_batch_sizes = np.log2(batch_sizes)

plt.figure(figsize=(10, 6))
bar_width = 0.5
bars = plt.bar(log_batch_sizes, times, width=bar_width, color='skyblue', edgecolor='black')

plt.xticks(log_batch_sizes, labels=[str(i) for i in batch_sizes])
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Time')
plt.title('Time vs Batch Size')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (10 if height >= 0 else -20),
        f'{height}',
        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9
    )

plt.savefig("better_bar_plot.png")

plt.show()
