import pandas as pd

# Load the datasets
true_news_df = pd.read_csv('True.csv')
false_news_df = pd.read_csv('Fake.csv')

# Add a 'target' column for both dataframes
true_news_df['target'] = 1  # 1 for true news
false_news_df['target'] = 0  # 0 for false news

# Concatenate the two dataframes
merged_df = pd.concat([true_news_df, false_news_df], ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_news.csv', index=False)

# We are gonna do EDA on this in the training.py file and there will be a new
# file named merged_news_2.csv which will have cleaned data !