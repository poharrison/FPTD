usr_input = input("Type something: ")
print(usr_input)


def get_bin_lists(bins_list_name):
    while True:
        try:
            bins_list = list(int(x) for x in input("Please list the " + bins_list_name + " (Ex. 1, 2, 3, 4): ").split(","))
            return bins_list
        except ValueError:
            print("Please check the formatting of your input. Input should be a list of of numbers separated by commas.")


TARGET_BINS = get_bin_lists("target bins")

for x in TARGET_BINS:
    print(x)