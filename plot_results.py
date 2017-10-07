from matplotlib import pyplot as plt
import pickle


LINEWIDTH=2.0
LINESTYLES=['-', '--', ':', '-.']

class MyException(Exception):
    pass


def plot_results(results, dirname, datatype="keras"):
    plt.close('all')

    fig, axarr = plt.subplots(2, sharex=True, sharey=True)

    if datatype == "keras":
        i = 0
        for k, result in results.items():
            axarr[0].plot(result.history['loss'], label=k, linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            axarr[1].plot(result.history['val_loss'], label=k, linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            i += 1
            i = i % len(LINESTYLES)
    elif datatype == "saved":
        for i, result in enumerate(results):
            i = i % len(LINESTYLES)
            axarr[0].plot(result['train_loss_history'], label=result['label'], linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            axarr[1].plot(result['val_loss_history'], label=result['label'], linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
    else:
        raise MyException("Illegal Datatype")
    axarr[1].set_xlabel('Epoch')
    axarr[0].set_ylabel('Training Error')
    axarr[1].set_ylabel('Validation Error')
    handles, labels = axarr[0].get_legend_handles_labels()
    lgd1 = plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.2))
    plt.gcf().tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    axarr[0].grid()
    axarr[1].grid()
    plt.savefig('plots/' + dirname[:-1] + "_val.png", bbox_extra_artists=(lgd1, ), bbox_inches='tight')


def save_curve_data(results, filename):
    to_save = []
    for k, result in results.items():
        to_save.append(
            {"label": k, "train_loss_history": result.history['loss'], "val_loss_history": result.history['val_loss']})
    with open(filename, "wb") as f:
        pickle.dump(to_save, f)


if __name__=="__main__":
    print("Input Pickled Data Path: ")
    dirname = input()
    with open(dirname, "rb") as f:
        data = pickle.load(f)
    dirname = dirname.replace("/", "_")
    plot_results(data, dirname.replace("\/", "_"), datatype="saved")