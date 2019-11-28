# Copyright @ University of Essex, 2019
# Author: Somdip Dey

def get_score(abstract_input):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    import numpy as np
    import pandas as pd
    from csv import reader,writer
    import operator as op
    import string
    from sklearn import neighbors

    #Read data from corpus
    r = reader(open('one100words.csv','r'))
    abstract_list = []
    score_list = []
    institute_list = []
    row_count = 0
    for row in list(r)[1:]:
        institute,score,abstract = row[0], row[1], row[2]
        if len(abstract.split()) > 0:
          institute_list.append(institute)
          score = float(score)
          if score >= 3.2:
              score = 1
          elif (score >= 3.0 and score < 3.2):
              score = 2
          elif (score >= 2.8 and score < 3.0):
              score = 3
          elif (score >= 2.5 and score < 2.8):
              score = 4
          elif (score >= 2.2 and score < 2.5):
              score = 5
          elif (score >= 2.0 and score < 2.2):
              score = 6
          elif (score >= 1.5 and score < 2.0):
              score = 7
          elif (score >= 1.0 and score < 1.5):
              score = 8
          score_list.append(score)
          abstract = abstract.translate(string.punctuation).lower()
          abstract_list.append(abstract)
          row_count = row_count + 1

    #print("Total processed data: ", row_count)
    total_processed_data = row_count

    #Vectorize (TF-IDF, ngrams 1-4, no stop words) using sklearn -->
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,4),
                         min_df = 0, stop_words = 'english', sublinear_tf=True)
    response = vectorizer.fit_transform(abstract_list)
    classes = score_list
    feature_names = vectorizer.get_feature_names()

    # Initialize model
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(response, classes)

    # Get prediction for the input abstract
    out_response = vectorizer.transform([abstract_input])
    predicted = clf.predict(out_response)

    # Get score for predict class
    if int(predicted[0]) == 1:
        score_output = 3.2
    elif int(predicted[0]) == 2:
        score_output = 3.0
    elif int(predicted[0]) == 3:
        score_output = 2.8
    elif int(predicted[0]) == 4:
        score_output = 2.5
    elif int(predicted[0]) == 5:
        score_output = 2.2
    elif int(predicted[0]) == 6:
        score_output = 2.0
    elif int(predicted[0]) == 7:
        score_output = 1.5
    elif int(predicted[0]) == 8:
        score_output = 1.0

    return score_output, total_processed_data

def main():
    abstract_input = "This paper is the first in membrane computing to propose a systematic way of deriving model checking specifications and test sets for basic classes of P systems. The formal verification method introduced in the paper has led to further research regarding the verification and testing of various classes of P systems. This research resulted from a Romanian Research Council project (PN-II-ID-PCE-2011-3-0688) between Bucharest and Sheffield, and led to the building of formal verification tools in the 4 partner interdisciplinary EPSRC funded RoadBlock project (Gheorghe PI). [15 GoogleScholar citations]"

    score, total_processed_data = get_score(abstract_input)

    print("Expected score: " + str(score))
    print("Total processed data: " + str(total_processed_data))

if __name__ == '__main__':
    main()
