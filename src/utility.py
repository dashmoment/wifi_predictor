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