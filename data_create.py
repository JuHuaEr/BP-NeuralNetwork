import matplotlib.pyplot as plt
import numpy as np


class DataCreater:
    def __init__(self, num):
        self.num = num
        self.x = np.arange(-8, 8, 16 / num)
        self.y = self.x*np.cos(self.x)

    def normliza_x(self):
        self.x = self.x / max(self.x)

    def normlize_xy(self):
        self.x = self.x / max(self.x)
        self.y = self.y / max(self.y)

    def get_data(self):
        return (self.x, self.y)

    def plot_data(self):
        plt.scatter(self.x, self.y, color='b')
        plt.plot(self.x, self.y, color='b')
        plt.show()


if __name__ == '__main__':
    demo = DataCreater(15)
    demo.plot_data()
    demo.normliza_x()
    print(demo.get_data())
