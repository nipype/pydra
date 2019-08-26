import os


def saving(filename):
    with open(filename, "w") as f:
        f.write("Hello!")
    print(filename)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-f", dest="filename", help="filename")
    args = parser.parse_args()

    saving(args.filename)
