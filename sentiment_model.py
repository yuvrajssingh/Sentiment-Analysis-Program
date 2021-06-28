import turicreate as tc

# loading data set
sf = tc.SFrame('Reviews.csv')

# cleaning data
sf = sf.remove_columns(['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Id', 'ProductId', 'UserId', 'ProfileName', 'Summary'])
sf['word_count'] = tc.text_analytics.count_words(sf['Text'])
sf = sf[sf['Score'] != 3]
sf['sentiment'] = sf['Score'] >= 4

# creating model
train_data, test_data = sf.random_split(0.8, seed=0)
sentiment_model = tc.logistic_classifier.create(train_data,
                                                target='sentiment',
                                                features=['word_count'],
                                                validation_set=test_data)

# saving model
sentiment_model.save('my_model')
