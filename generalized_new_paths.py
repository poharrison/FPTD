def get_bin_lists(bins_list_name):
    while True:
        try:
            bins_list = list(int(x) for x in raw_input("Please list the %s (Ex. 1, 2, 3, 4): " % bins_list_name).split(","))
            return bins_list
        except ValueError:
            print("Please check the formatting of your input. Input should be a list of of numbers separated by commas.")


def get_nbins():
    while True:
        try:
            NBINS = int(input("Total number of bins: "))
            return NBINS
        except (NameError, TypeError):
            print("Please input a single number.")


NBINS = get_nbins()
INIT_BINS = get_bin_lists("initial bins")
if INIT_BINS.length > NBINS:
    print("Cannot have more initial bins than bins total.")
    INIT_BINS = get_bin_lists("initial bins")
TARGET_BINS = get_bin_lists("target bins")
if TARGET_BINS.length > NBINS:
    print("Cannot have more target bins than bins total.")
    INIT_BINS = get_bin_lists("target bins")

for x in TARGET_BINS:
    print(x)