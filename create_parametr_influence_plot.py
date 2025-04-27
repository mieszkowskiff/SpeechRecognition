import matplotlib.pyplot as plt
import numpy as np
from split_dataset import class_list

f1s = np.array([
    [0.826, 0.861, 0.875, 0.873],
    [0.846, 0.846, 0.806, 0.806],
    [0.838, 0.827, 0.806, 0.806]
])

parameters = np.array([
    [76108, 226764, 377420, 528076],
    [208716, 624588, 1040460, 1456332],
    [341324, 1022412, 1703500, 2384588]

])

plt.figure(figsize=(9, 9))
plt.imshow(f1s, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(f1s.shape[0]):
    for j in range(f1s.shape[1]):
        plt.text(j, i, "overfit" if (i == 2 and j == 2 )or (i == 2 and j == 3) or (i == 1 and j == 3) else f"{f1s[i, j]:.3f}",
                 horizontalalignment="center",
                 color="white" if i == 0 and (j == 2 or j == 3) else "black",
                 fontsize=20)
        plt.text(j, i + 0.35, f"{parameters[i, j]:,}".replace(",", " "),
                    horizontalalignment="center",
                    color= "white" if i == 0 and (j == 2 or j == 3) else "black",
                    fontsize=12)
        
plt.title('F1 Score Matrix', fontsize=20)
plt.ylabel('#Heads', fontsize=20)
plt.xlabel('#Blocks', fontsize=20)

plt.xticks(range(4), [2, 6, 10, 14], fontsize = 16)
plt.yticks(range(3), [2, 6, 10], fontsize = 16)
plt.tight_layout()
plt.savefig(f"./images/f1s_matrix.png")

