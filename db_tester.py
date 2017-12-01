import sqlite3

subreddit_names_to_follow = [ 'getmotivated', 'adviceanimals',
                             'europe']

with sqlite3.connect('reddit.db') as conn:


    res = conn.execute('''delete
     from preprocessed_comments''').fetchall()
    conn.commit()

    res = conn.execute('''SELECT *
     from preprocessed_comments''').fetchall()
    print(len(res))
