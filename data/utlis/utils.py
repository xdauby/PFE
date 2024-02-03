
import json
import matplotlib.pyplot as plt

def plot_errors(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    train_errors = data["train"]
    val_errors = data["val"]

    epochs = range(1, len(train_errors) + 1)

    plt.plot(epochs, train_errors, label='Training Error')
    plt.plot(epochs, val_errors, label='Validation Error')
    
    plt.title('Training and Validation Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    
    plt.show()


#json_file_path = 'data/errors/metainvnet_3e5_tv_0_005_200_l2_loss'
#plot_errors(json_file_path)