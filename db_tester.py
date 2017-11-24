import sqlite3

subreddit_names_to_follow = [ 'getmotivated', 'adviceanimals',
                             'europe']

with sqlite3.connect('reddit.db') as conn:


    res = conn.execute('''select *
     from preprocessed_comments''').fetchall()
    print(len(res))
