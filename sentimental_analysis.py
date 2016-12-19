from sqlite3 import connect
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

stops = set(stopwords.words("english"))


def clean_tweet(tweet):
    text = re.sub('\s+', ' ', tweet[0])
    return ((' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())).lower(), tweet[1])


def remove_stopwords(tweet):
    tweet_words = tweet[0].split()
    meaningful_words = [w for w in tweet_words if not w in stops]
    return (" ".join(meaningful_words), tweet[1])


def sanitize_tweets(tweets):
    clean_train_tweets = []

    for tweet in tweets:
        clean_train_tweets.append(remove_stopwords(clean_tweet(tweet)))

    return clean_train_tweets


def main():
    connection = connect('data/database.sqlite')
    
    cursor = connection.cursor()
    cursor.execute("SELECT text, airline_sentiment FROM Tweets")

    connection.close()

    clean_tweets = sanitize_tweets(cursor.fetchall())
    train, test = train_test_split(clean_tweets, train_size=0.5)

    vectorizer = CountVectorizer(
        analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    train_tweets = [tweet[0] for tweet in train]
    train_sentiment = [tweet[1] for tweet in train]

    train_data_features = vectorizer.fit_transform(train_tweets)
    train_data_features = train_data_features.toarray()

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train_sentiment)

    test_tweets = [tweet[0] for tweet in test]
    test_sentiment = [tweet[1] for tweet in test]

    test_data_features = vectorizer.transform(test_tweets)
    test_data_features = test_data_features.toarray()

    result = forest.predict(test_data_features)

    print accuracy_score(test_sentiment, result)  # 0.752450641999


if __name__ == "__main__":
    main()
