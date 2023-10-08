import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


classes = ["0_background", "1_trash", "2_paper", "3_plastic", "4_metal", "5_electronic_invoice", "6_bubble_wrap", "7_thin_plastic_bag", "8_fruit_mesh_bag", "9_thin_film_paper_cup"]
classes_official = ["Background", "Trash", "Paper", "Plastic", "Metal", "Electronic Invoice", "Bubble Wrap", "Thin Plastic Bag", "Fruit Mesh Bag", "Thin Film Paper Cup"]


def load_matrix_from_txt(filename):
    matrix = np.loadtxt(filename)
    return matrix.astype(int)

filename = '.\\evaluation\\conf_matrix_V9.txt'
loaded_matrix = load_matrix_from_txt(filename)
print(loaded_matrix)

# loaded_matrix =  [[13, 0,  0,  0,  0,  0,  0,  0,  0,  0]
# ,[ 0, 47,  1,  1,  0,  1,  0,  0,  0,  0]
# ,[ 0,  0, 50,  0,  0,  0,  0,  0,  0,  0]
# ,[ 0,  0,  0, 37,  0,  0,  0,  0,  0,  0]
# ,[ 0,  0,  0,  0, 50,  0,  0,  0,  0,  0]
# ,[ 0,  0,  0,  0,  0, 20,  0,  0,  0,  0]
# ,[ 0,  0,  0,  2,  0,  0, 18,  0,  0,  0]
# ,[ 0,  1,  0,  0,  0,  0,  0, 19,  0,  0]
# ,[ 0,  0,  0,  0,  0,  0,  0,  0, 15,  0]
# ,[ 0,  0,  0,  0,  0,  0,  0,  0,  0, 20]]

conf_matrix = []
for row in loaded_matrix:
    new_row = list(row / float(sum(row)))
    conf_matrix.append(new_row)

print(conf_matrix)


plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar(shrink=0.9)
plt.xlabel('Predicted',fontsize=14)
plt.ylabel('Actual',fontsize=14)
tick_marks = np.arange(10)
plt.xticks(tick_marks, classes_official,rotation=30)
plt.yticks(tick_marks, classes_official)

for i in range(10):
    for j in range(10):
        plt.text(j, i, str(format(round(conf_matrix[i][j],3),'.2f')), ha='center', va='center', color='black', fontsize=8)

plt.show()
plt.savefig('confusion_matrix_V8.png', format='png')