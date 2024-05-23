from sqlalchemy import create_engine, text
import pandas as pd

# Database connection
db_url = db_url  # Make sure to replace this with your actual DB URL
engine = create_engine(db_url)

# Fetch comments needing new labeling
query = "SELECT username, comments FROM public.reddit_usernames_comments WHERE label IS NULL;"
comments_df = pd.read_sql(query, engine)

# Function to determine label based on reason
def determine_label(comment):
    comment_lower = comment.lower()  # Normalize the text to lower case for consistency
    if 'vet' in comment_lower or 'veterinary' in comment_lower:
        return 'Veterinarian'
    elif 'doctor' in comment_lower or 'medical' in comment_lower:
        if 'resident' in comment_lower or 'residency' in comment_lower:
            return 'Other'
        else:
            return 'Medical Doctor'
    return 'Other'

# Apply function to determine new labels
comments_df['new_label'] = comments_df['comments'].apply(determine_label)

# Update database with new labels
with engine.connect() as connection:
    transaction = connection.begin()  # Start a transaction
    try:
        for index, row in comments_df.iterrows():
            update_query = text("""
                UPDATE public.reddit_usernames_comments
                SET label = :new_label
                WHERE username = :username AND comments = :comments
            """)
            connection.execute(update_query, {'new_label': row['new_label'], 'username': row['username'], 'comments': row['comments']})
        transaction.commit()  # Commit the transaction
        print("Database update was successful.")
    except Exception as e:
        transaction.rollback()  # Roll back the transaction on error
        print(f"An error occurred: {e}")
