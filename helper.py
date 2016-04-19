import argparse

def main():
    print
    parser = argparse.ArgumentParser(description = "CS584 Machine Learning Project ", usage=readme())

    args = parser.parse_args()

    print "\n\nDigit and Letter: Recognition and Classification."
    print

def readme():
    with open("helper.txt") as myfile:
        data = myfile.read().replace('\n', '\n')
    return data

if __name__ == "__main__":
    main()

