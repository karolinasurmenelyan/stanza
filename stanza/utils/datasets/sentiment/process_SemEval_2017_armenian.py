import os
import stanza
from stanza.utils import default_paths
from stanza.utils.datasets.sentiment.process_utils import SentimentDatum, write_dataset


def convert_label(label):
    """
    negative/positive or error
    """
    if label == "negative":
        return 0
    if label == "positive":
        return 1
    raise ValueError("Unexpected label %s" % label)


def tokenize(sentiment_data, pipe):
    """
    Takes a list of (label, text) and returns a list of SentimentDatum with tokenized text
    Only the first 'sentence' is used - ideally the pipe has ssplit turned off
    """
    docs = [x.text for x in sentiment_data]
    in_docs = [stanza.Document([], text=d) for d in docs]
    out_docs = pipe(in_docs)

    sentiment_data = [SentimentDatum(datum.sentiment,
                                     [y.text for y in doc.sentences[0].tokens])  # list of text tokens for each doc
                      for datum, doc in zip(sentiment_data, out_docs)]

    return sentiment_data


def read_file(filename, pipe):
    """
    Read and tokenize set
    """
    sentiment_data = []

    with open(filename, encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if line_idx == 0:
                continue
            pieces = line.split(',', maxsplit=4)
            if len(pieces) != 5:
                raise ValueError(
                    "Unexpected format at line %d: all label lines should be len==5\n%s" % (line_idx, line))
            if pieces[4].startswith('"'):
                pieces[4] = pieces[4].split('"', maxsplit=1)[1]
                pieces[4] = pieces[4].rsplit('"', maxsplit=1)[0]
            text, label = pieces[4], pieces[3]
            try:
                label = convert_label(label)
            except ValueError:
                raise ValueError("Unexpected train label %s at line %d\n%s" % (label, line_idx, line))
            sentiment_data.append(SentimentDatum(label, text))

    print("Read %d texts from %s" % (len(sentiment_data), filename))
    sentiment_data = tokenize(sentiment_data, pipe)
    return sentiment_data

def convert_SemEval_2017(in_directory, out_directory, dataset_name):
    """
    Read all of the data from in_directory/armenian/dataset_name, write it to out_directory...
    """
    in_directory = os.path.join(in_directory, "armenian", "SemEval2017Task4translated")

    pipe = stanza.Pipeline(lang="hy", processors="tokenize", tokenize_no_ssplit=True)

    test_filename = os.path.join(in_directory, "SentimentTest_B.csv")
    test = read_file(test_filename, pipe)

    dev_filename = os.path.join(in_directory, "SentimentDev_B.csv")
    dev = read_file(dev_filename, pipe)

    train_filename = os.path.join(in_directory, "SentimentTrain_B.csv")
    train = read_file(train_filename, pipe)

    print("Total train items: %8d" % len(train))
    print("Total dev items:   %8d" % len(dev))
    print("Total test items:  %8d" % len(test))

    write_dataset((train, dev, test), out_directory, dataset_name)


def main(paths):

    in_directory = os.getenv('SENTIMENT_BASE')
    out_directory = os.getenv('SENTIMENT_DATA_DIR')

    convert_SemEval_2017(in_directory, out_directory, "hy_SemEval2017Task4translated")


if __name__ == '__main__':
    paths = default_paths.get_default_paths()
    main(paths)
