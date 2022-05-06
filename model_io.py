import os
import matplotlib.pyplot as plt

import plot_func
import model

def save_snapshot(model_components, model_path, snapshot):

    groups, inputs, connections = model_components

    path = f"{model_path}/{snapshot}/"
    os.mkdir(path)

    #Plotting performance, for windowlength see plot_func.py
    plot_func.plot_decoding_performance(inputs.values())
    plt.savefig(f"{path}/decoding_performance.svg", dpi=model.model_dpi)

    # MOUSE MODEL SPECIFIC
    plot_func.plot_mouse_experiment(model_components)
    plt.savefig(f"{path}/mismatch.svg", dpi=model.model_dpi)

    for group in groups.values():
        group.save(path)
    plt.close('all')

    for inp in inputs.values():
        inp.save(path)
    plt.close('all')

    for connection in connections.values():
        connection.save(path)
    plt.close('all')
