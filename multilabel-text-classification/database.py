from sqlalchemy import create_engine
import pandas as pd

# Database connection string
#db_url = "postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false"
db_url = "postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist"

engine = create_engine(db_url)


query = "SELECT username, comments FROM public.reddit_usernames_comments"
comments_df = pd.read_sql(query, engine)
comments_df.head


from sqlalchemy import create_engine, MetaData, Table

# Database connection
db_url = db_url  # Replace with your actual database URL
engine = create_engine(db_url)
metadata = MetaData()

# Reflect the table
table = Table('reddit_usernames_comments', metadata, autoload_with=engine)

# Print the column names in the list comprehension
print("Column Names:", [column.name for column in table.columns])