import numpy as np
import matplotlib.pyplot as plt


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)




f = open("age_alexnet_train_loss.txt", "r")
restlos=f.readlines()
restlos = [float(i) for i in restlos]

destlos=[]
f = open("age_alexnet_val_loss.txt", "r")
destlos=f.readlines()
destlos = [float(i) for i in destlos]
f.close()
epoch = range(1, 11)
def main():
    ax1 = plt.subplot(212)
    ax1.margins(0.1)           # Default margin is 0.05, value 0 means fit


    ax1.text(-0.059, 0.04, r'Best in Epoch 7', fontsize=13)
    ax1.text(-0.059, 0.02, r'Validation Loss = 0.4376', fontsize=13)
    ax1.text(-0.059, 0.0, r'Validation Accuracy = 79.50%', fontsize=13)

    ax2 = plt.subplot(221)
            # Values >0.0 zoom out
    ax2.plot(epoch,restlos)
    ax2.plot(epoch,destlos)
    ax2.set_title('Loss of AlexNet')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['Training', 'Validation'], loc='upper right')

    ax3 = plt.subplot(222)
      # Values in (-0.5, 0.0) zooms in to center
    ax3.plot(epoch,destlos)
    ax3.plot(epoch,restlos)

    ax3.set_title('Zoomed in')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend(['Training', 'Validation'], loc='upper right')

    #plt.show()
main()