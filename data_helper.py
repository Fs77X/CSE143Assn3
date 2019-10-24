
import re


def read_file(fname):
    with open(fname, "rb") as fin:
        raw_data = fin.read().decode("latin1")
    return raw_data


def get_score(review):
    """
    This function extracts the integer score from the review.

    Write a regular expression that searches for the Overall score
    and then extract the score number.

    :param review: All text associated with the review.
    :return: int: score --- the score of the review
    """
    # YOUR CODE GOES HERE
    return int(re.search(r'Overall = ([0-9])', review).group(1))


def get_text(review):
    """
    This function extracts the description part of the
    imdb review.

    Use regex to extract the Text field of the review,
    similar to the get_score() function.

    :param review:
    :return: str: text -- the textual description part of the imdb review.
    """

    # YOUR CODE GOES HERE
    return re.search(r'Text = "(.*)"', review).group(1)


def get_reviews(raw_data):
    """
    Process the imdb review data. Split the data into two
    lists, one list for positive reviews and one list for negative
    reviews. The list items should be the descriptive text of
    each imdb review.

    A positive review has a overall score of at least 3 and
    negative reviews have scores less than 3.

    :param raw_data:
    :return:
    """
    # print(raw_data)
    positive_texts = []
    negative_texts = []

    for review in re.split(r'\.\n', raw_data):
        if review:
            overall_score = get_score(review)
            # print(overall_score)
            review_text = get_text(review)
            # print(review)

            # YOUR CODE GOES HERE
            if overall_score > 5:
                positive_texts.append(review_text)
            elif overall_score < 5:
                negative_texts.append(review_text)
            


    return positive_texts, negative_texts




def test_main():
    datafile = "imdb-training.data"
    raw_data = read_file(datafile)
    p, n = get_reviews(raw_data)

    assert p[0].startswith("If you loved Long Way Round you will enjoy this nearly as much."), p[0]
    assert n[0].startswith("How has this piece of crap stayed on TV this long?"), n[0]



if __name__ == "__main__":
    test_main()
