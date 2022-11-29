import matplotlib.pyplot as plt

def bar_viusualize(ammount,area,description):
    print("\n> Generating bar graph")
    count = ammount.value_counts()
    count.plot.bar()
    plt.ylabel(description)
    plt.xlabel(area)
    plt.show()
    print("Bar graph visualization ended<\n")

if __name__ == '__main__':
    bar_viusualize()