
import nltk
import re
import word_category_counter
import data_helper
import os
import sys
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.util import bigrams, ngrams
from contextlib import redirect_stdout


DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)


def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token.lower()

    else:

        # YOUR CODE GOES HERE
        normalized_token = []
        word = token.lower()
        stop = stopwords.words('english')
        if word not in stop:
            normalized_token = re.findall('\w+', word)
            if normalized_token:
                return word
            else:
                return None
        else:
            normalized_token = None



    return normalized_token


def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """

    words = []
    tags = []

    # tokenization for each sentence

    # YOUR CODE GOES HERE

    if should_normalize:
        sent = nltk.sent_tokenize(text)
        for sentence in sent:
            leTags = nltk.pos_tag(nltk.word_tokenize(sentence))
            for word in leTags:
                # print(word)
                norm = normalize(word[0])
                # print(norm)
                if norm:
                    text = ''
                    for wrd in norm:
                        text = text + wrd
                    

                    words.append(text)
                tags.append(word[1])

    else:
        sent = nltk.sent_tokenize(text)
        for sentence in sent:
            leTags = nltk.pos_tag(nltk.word_tokenize(sentence))
            for word in leTags:
                words.append(word[0])
                tags.append(word[1])



    return words, tags


def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """



    feature_vectors = {}
    feature_vectors.update({'UNI_way': 0.0125})
    feature_vectors.update({'UNI_one': 0.030303030303030304})
    feature_vectors.update({'UNI_movie': 0.0273972602739726})
    feature_vectors.update({'UNI_bit': 0.013513513513513514})
    feature_vectors.update({'UNI_fact': 0.015151515151515152})
    feature_vectors.update({'UNI_yet': 0.017241379310344827})
    feature_vectors.update({'UNI_may': 0.0125})
    feature_vectors.update({'UNI_times': 0.013513513513513514})
    feature_vectors.update({'UNI_new': 0.013513513513513514})
    feature_vectors.update({'UNI_pretty': 0.01639344262295082})
    feature_vectors.update({'UNI_something':0.013})
    feature_vectors.update({'UNI_shot': 0.014})
    feature_vectors.update({'UNI_nice': 0.012})
    feature_vectors.update({'UNI_without': 0.013})

    

    # # YOUR CODE GOES HERE
    # # print('bruh')
    unigrams = tokens
    # l = len(tokens)
    bigG = list(bigrams(tokens))
    trigrams = list(ngrams(tokens, 3))


    fDistUni = FreqDist(unigrams)
    fDistUni = fDistUni.most_common()
    fDistBi = FreqDist(bigG)
    fDistBi = fDistBi.most_common()
    fDistTri = FreqDist(trigrams)
    fDistTri = fDistTri.most_common()

    for pair in fDistUni:
        # print(pair)
        feature_vectors.update({('UNI_'+pair[0]) : pair[1]/len(fDistUni)})
   
    
    for pair in fDistBi:
        # print(pair)
        feature_vectors.update({'BI_'+pair[0][0] + '_' + pair[0][1] : pair[1]/len(fDistBi)})
    
    for pair in fDistTri:
        # print(pair)
        feature_vectors.update({'TRI_' + pair[0][0] + '_' + pair[0][1] + '_' + pair[0][2] :pair[1]/len(fDistTri)})


    return feature_vectors


def get_pos_features(tags):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}
    feature_vectors.update({'BI_VBG_NN': 0.025})
    feature_vectors.update({'UNI_CD': 0.25})
    feature_vectors.update({'BI_JJ_RB': 0.02702702702702703})
    feature_vectors.update({'UNI_RB' : 0.7272727272727273})
    feature_vectors.update({'BI_NNS_VBP': 0.05})
    feature_vectors.update({'UNI_VBP': 0.2222222222222222})

    # # YOUR CODE GOES HERE
    unigrams = tags
    bigG = list(bigrams(tags))
    trigram = list(ngrams(tags, 3))

    fDistUni = FreqDist(unigrams)
    # fDistUni = fDistUni.most_common()
    fDistBi = FreqDist(bigG)
    # fDistBi = fDistBi.most_common()
    fDistTri = FreqDist(trigram)
    # fDistTri = fDistTri.most_common()

    for pair in fDistUni:
        feature_vectors.update({('UNI_'+pair) : fDistUni[pair]/len(fDistUni)})
    
    for pair in fDistBi:
        feature_vectors.update({'BI_'+pair[0] + '_' + pair[1] : fDistBi[pair]/len(fDistBi)})
    
    for pair in fDistTri:
        feature_vectors.update({'TRI_' + pair[0] + '_' + pair[1] + '_' + pair[2] :fDistTri[pair]/len(fDistTri)})



    return feature_vectors



def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    # TODO: binning

    feature_vectors = {}
    feature_vectors.update({'Insight': 5})
    feature_vectors.update({'Positive Emotion': 10})
    feature_vectors.update({'Discrepancy': 3})
    feature_vectors.update({'Discrepancy': 6})
    feature_vectors.update({'Tentative': 6})
    feature_vectors.update({'Negative Emotion': 5})
    feature_vectors.update({'Positive Emotion': 7})
    feature_vectors.update({'Positive Emotion': 11})
    feature_vectors.update({'Discrepancy': 2})
    feature_vectors.update({'Discrepancy': 4})
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)

    # # All possible keys to the scores start on line 269
    # # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]
    anger_score = liwc_scores['Anger']
    insight_score = liwc_scores['Insight']
    sadness_score = liwc_scores['Sadness']
    discrepancy_score = liwc_scores['Discrepancy']
    tentative_score = liwc_scores['Tentative']
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score
    feature_vectors['Anger'] = anger_score
    feature_vectors['Insight'] = insight_score
    feature_vectors['Discrepancy'] = discrepancy_score
    feature_vectors['Sadness'] = sadness_score
    feature_vectors['Tentative'] = tentative_score


    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1
    
    if anger_score > sadness_score:
        feature_vectors['liwc:anger'] = 1
    else:
        feature_vectors['liwc:sadness'] = 1

    if insight_score > discrepancy_score:
        feature_vectors['liwc:insight'] = 1
    else:
        feature_vectors['liwc:discrepancy'] = 1
    if tentative_score > discrepancy_score:
        feature_vectors['liwc:tentative'] = 1
    else: 
        feature_vectors['liwc:discrepancy'] = 1
    return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"}


def get_opinion_features(tags):
    """
    This function creates the opinion lexicon features
    as described in the assignment3 handout.

    the negative and positive data has been read into the following lists:
    * neg_opinion
    * pos_opinion

    if you haven't downloaded the opinion lexicon, run the following commands:
    *  import nltk
    *  nltk.download('opinion_lexicon')

    :param tags: tokens
    :return: feature_vectors: a dictionary values for each opinion feature
    """
    neg_opinion = opinion_lexicon.negative()
    pos_opinion = opinion_lexicon.positive()
    feature_vectors = {}

    # YOUR CODE GOES HERE
    feature_vectors.update({'UNI_POS_pretty': 0.01639344262295082})
    feature_vectors.update({'UNI_POS_well': 0.013513513513513514})
    feature_vectors.update({'UNI_POS_great': 0.023809523809523808})
    feature_vectors.update({'UNI_POS_good': 0.03225806451612903})
    feature_vectors.update({'UNI_POS_like': 0.016666666666666666})
    feature_vectors.update({'UNI_NEG_unexpectedly': 0.0125})
    feature_vectors.update({'UNI_POS_perfectly': 0.015151515151515152})
    feature_vectors.update({'UNI_POS_thank': 0.016666666666666666})
    feature_vectors.update({'UNI_POS_clearly': 0.013513513513513514})
    feature_vectors.update({'UNI_NEG_confusing': 0.013513513513513514})
    words = tags
    wordF = FreqDist(words)
    for word in neg_opinion:
        if wordF.freq(word) > 0.0:
            feature_vectors.update({'UNI_NEG_' + word : wordF[word]/len(wordF)})

    for word in pos_opinion:
        if wordF.freq(word) > 0.0:
            feature_vectors.update({'UNI_POS_' + word : wordF[word]/len(wordF)})





    return feature_vectors


def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)
    for category in category_text_dict:
  
        for text in category_text_dict[category]:
         

            words, tags = get_words_tags(text)
            feature_vectors = {}

            # YOUR CODE GOES HERE
            if feature_set == "word_features":
                feature_vectors.update(get_ngram_features(words))
                # do this
            elif feature_set == "word_pos_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))

                # print('wip')
                # do this
            elif feature_set == "word_pos_liwc_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))
                feature_vectors.update(get_liwc_features(words))
                # print('wip')
                # do this
            elif feature_set == "word_pos_opinion_features":
                feature_vectors.update(get_ngram_features(words))
                feature_vectors.update(get_pos_features(tags))
                feature_vectors.update(get_opinion_features(words))
                # do this
            

            features_category_tuples.append((feature_vectors, category))
            
            all_texts.append(text)

    return features_category_tuples, all_texts


def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write("{0:<10s}\t{1}\n".format(category, features))






def features_stub():
    datafile = "imdb-training.data"
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}
    # FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"}

    feature_set = "word_pos_opinion_features"

    features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)

    # raise NotImplemented
    filename = feature_set + "-testing-features.txt"
    write_features_category(features_category_tuples, filename)



if __name__ == "__main__":
    features_stub()
