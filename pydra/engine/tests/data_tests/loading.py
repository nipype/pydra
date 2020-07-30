import os


def loading(filename):
    with open(filename) as f:
        txt = f.read()
    print(txt)


if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-f", dest="filename", help="filename")
    args = parser.parse_args()

    loading(args.filename)
