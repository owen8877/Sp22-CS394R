import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]

with open(filename,'r') as f:
    lines = f.readlines()

    sample_average = {
        'average_rs': [float(n) for n in lines[0].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[1].strip().split()],
    }
    constant = {
        'average_rs': [float(n) for n in lines[2].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[3].strip().split()],
    }

    assert len(sample_average['average_rs']) == len(sample_average['average_best_action_taken']) == \
        len(constant['average_rs']) == len(constant['average_best_action_taken']) \
           # == 100

    fig,axes = plt.subplots(2,1)

    axes[1].set_ylim([0.,1.])

    axes[0].plot(sample_average['average_rs'], label='sample average')
    axes[1].plot(sample_average['average_best_action_taken'], label='sample average')

    axes[0].plot(constant['average_rs'], label='constant')
    axes[1].plot(constant['average_best_action_taken'], label='constant')

    plt.legend()
    fig.show()
    # _ = input()

