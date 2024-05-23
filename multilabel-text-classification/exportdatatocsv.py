from sqlalchemy import create_engine, MetaData, Table, select
import pandas as pd

# Database connection
db_url = db_url # Replace with your actual database URL
engine = create_engine(db_url)
metadata = MetaData()

# Reflect the table
table = Table('reddit_usernames_comments', metadata, autoload_with=engine)

# Select all data from the table
stmt = select(table)

# Execute the query and fetch all data
with engine.connect() as connection:
    result = connection.execute(stmt)
    data = result.fetchall()

# Create a DataFrame from the fetched data
df = pd.DataFrame(data, columns=result.keys())

# Save the DataFrame to CSV, column names will be included automatically
df.to_csv('reddit_usernames_comments.csv', index=False)  # Adjust path as needed

print("Data successfully saved to 'reddit_usernames_comments.csv'.")
