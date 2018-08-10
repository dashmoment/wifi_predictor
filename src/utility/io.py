import matplotlib.pyplot as plt
silence = False

def c_print(print_content, silent = silence):
    
    if silent == False:
        print(print_content)


def c_save_image(plot_func, *argw, filename='test.png',silent = silence):
    
    if silent == False:
        fig = plt.figure(figsize=(6, 6))
        plot_func(*argw)
        fig.savefig(filename, dpi=fig.dpi)


def show_train_history(train_history):

    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])

    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../analysis_result_rahul/train_history.png')
