import numpy as np

CLASS_LABEL_PATH='./synset.txt'


def do_softmax(outputs):
    # Softmax
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    return outputs

def show_top5(outputs):
    with open(CLASS_LABEL_PATH, 'r') as f:
        labels = [l.rstrip() for l in f]

    output = outputs[0].reshape(-1)
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'model: shufflenetv2_x1\n-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:03d}]: score={:.6f} class="{}"\n'.format(index, value, labels[index])
        else:
            topi = '[ -1]: score=0.0\n'
        top5_str += topi
    print(top5_str)
