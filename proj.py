import turicreate as tc

# loading model
sentiment_model = tc.load_model('my_model')

# taking input
review = input('\nEnter your review: ')

# processing input
x = tc.SFrame({'Text': [review]})
x['word_count'] = tc.text_analytics.count_words(x['Text'])

output_bool = sentiment_model.predict(x)
output_prob = sentiment_model.predict(x, output_type='probability')

# output prediction
if output_bool[0] == 1:
    print(f"\nYour review is POSITIVE!")
    print(f"Probability = {output_prob[0] * 100} %")
else:
    print(f"\nYour review is NEGATIVE!")
    print(f"Probability = {output_prob[0] * 100} %")