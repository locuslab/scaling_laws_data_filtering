import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

keys_of_interest = ["imagenet1k", "cifar10", "vtab/caltech101", "vtab/cifar100", "food101", "imagenet_sketch", "imagenetv2", "imagenet-a", "imagenet-o", "imagenet-r", "objectnet", "vtab/flowers", "vtab/pets", "voc2007", "vtab/resisc45", "cars", "retrieval/flickr_1k_test_image_text_retrieval", "retrieval/mscoco_2014_5k_test_image_text_retrieval"]
# for these keys, make a clean names dict
clean_dataset_names = {
    "imagenet1k": "ImageNet-1k",
    "cifar10": "CIFAR-10",
    "vtab/caltech101": "Caltech101",
    "vtab/cifar100": "CIFAR100",
    "food101": "Food101",
    "imagenet_sketch": "ImageNet-Sketch",
    "imagenetv2": "ImageNetV2",
    "imagenet-a": "ImageNet-A",
    "imagenet-o": "ImageNet-O",
    "imagenet-r": "ImageNet-R",
    "objectnet": "ObjectNet",
    "vtab/flowers": "Flowers",
    "vtab/pets": "Pets",
    "voc2007": "VOC2007",
    "vtab/resisc45": "RESISC45",
    "cars": "CARS",
    "retrieval/flickr_1k_test_image_text_retrieval": "Flickr1k",
    "retrieval/mscoco_2014_5k_test_image_text_retrieval": "MSCOCO",
    "18tasks": "Avg over 18 Tasks",
}

mpl.rcParams.update({
    # 'text.usetex': True,           # Use LaTeX for all text handling
    # 'font.family': 'serif',        # Use serif font instead of sans-serif
    'font.serif': 'Times',         # Specific serif font (e.g., Times)
    'axes.labelsize': 14,          # Size of axis labels
    'axes.titlesize': 16,          # Size of title
    'font.size': 14,               # Size of general text
    'legend.fontsize': 14,         # Size of legend text
    'xtick.labelsize': 14,         # Size of x-tick labels
    'ytick.labelsize': 12,         # Size of y-tick labels
    'figure.figsize': [6.4, 4.8],  # Default figure size
    'lines.linewidth': 1.5,        # Width of lines
    'lines.markersize': 6,         # Size of markers
    'axes.grid': True,             # Enable grid by default
    'grid.alpha': 0.5,             # Transparency of grid
    'grid.linestyle': '--',        # Style of grid lines
})

def plot_results(args, org_names, paths, x_vals_dict, y_vals_dict, error_vals_dict, fitted_vals_dict, a_values, b_values, c_values, d_values, samples_per_epoch):
    names = [org_names [i] + f"| b={b_values[i]:.2f} | $\\tau=${c_values[i]:.2f}" for i in range(len(org_names)-1)]
    names.append(org_names[-1] + f"\n| b={b_values[-1]:.2f} | $\\tau=${c_values[-1]:.2f}")
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'gray', 'pink', 'orange', 'purple', 'brown', 'gray']
    # colors = ['#008066', '#80c066', '#ffff66', '#67000d', '#0000FF']
    # colors = ['#004529', '#556b2f', '#8b0000', '#a50026', '#0000FF']
    colors = ['darkgreen', 'limegreen', 'darkorange', 'peru','red', 'dodgerblue']
    markers = ['o', 'x', '^', 's', 'p', '*', '+', 'D', 'v', 'h', '8', '1', '2', '3', '4', '5']

    # make two subplots in the same figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.3) 

    # now everyting about first subplot below
    for i, key in enumerate(paths.keys()):
        data_name = key
        x_vals = x_vals_dict[data_name]
        y_vals = y_vals_dict[data_name]
        error_vals = error_vals_dict[data_name]
        x_vals_millions = [i / 1e6 for i in x_vals]
        #increase line width to 3
        ax1.scatter(x_vals_millions, error_vals, label=names[i], color=colors[i], marker=markers[i], zorder = 10)
        fitted_deltas = fitted_vals_dict[data_name]
        fitted = fitted_deltas
        # make the lines ... dottes
        ax1.plot(x_vals_millions, fitted, color=colors[i], zorder = 1, linestyle='dotted', linewidth=3)
        if i == len(paths.keys()) - 1:
            # make thickness in legend 2 but not in plot
            # make legend colour green
            ax1.plot([], [], color="black", zorder = 0, linestyle='dotted', label="Fitted scaling curve for the pool")

        

    # ax1.ylabel("Imagenet Zero-Shot Error")
    data_name_clean = clean_dataset_names[args.metric]
    ax1.set_ylabel(f"{data_name_clean} Zero-Shot Error")
    # ax1.xlabel("Millions of Samples Seen")
    ax1.set_xlabel("Millions of Samples Seen")

    # make the legend location in the middle of third quadrant and and 
    # plt.legend(title='Legend Title')
    legend_title = "CLIP score based data pools"
    if args.filtering == "tmars":
        legend_title = "TMARS based data pools"
    #columsn = 2
    legend = ax1.legend(bbox_to_anchor=(-0.12, 1.45), loc='upper left', borderaxespad=0., title=legend_title, ncol=2)
    # legend = ax1.legend(loc='upper left', borderaxespad=0.)
    for text in legend.get_texts():
        if text.get_text() == 'Fitted scaling curve for the pool':
            text.set_color('black')

    # second subplot now
    for i, key in enumerate(paths.keys()):
        data_name = key
        initial_b = b_values[i]
        x_vals = np.arange(0, 10)
        y_vals = [-1*initial_b * (0.5**(j/c_values[i])) for j in x_vals]
        # make the lines ... dottes
        horizontal_line_key = "Top 10%|"
        if args.filtering == "clip":
            horizontal_line_key = "Top 10%-20%"
        if  horizontal_line_key in names[i]:
            #draw two horizontal lines one at x_vals = 2, and one at x_vals = 4, but it should be from x = 0 to x = 2 or 4 only
            ax2.plot([0, 2], [y_vals[2], y_vals[2]], color="black", zorder = 0, linestyle='-', linewidth=1.5)
            ax2.plot([0, 4], [y_vals[4], y_vals[4]], color="black", zorder = 0, linestyle='-', linewidth=1.5)
        ax2.plot(x_vals, y_vals, color=colors[i], zorder = 1, linestyle='dotted', linewidth=2, marker=markers[i], label = names[i])

    # legend = ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    # ax2.ylabel("Utility of b*\delta") #use latex
    ax2.set_ylabel("Data Utility ($b\\times\delta^{epoch}}$)")
    # ax2.xlabel("Number of Repetitions")
    ax2.set_xlabel("Number of Repetitions")
    plt.savefig(f"plots/{args.plot_name}.pdf", bbox_inches='tight')
