import sqlite3

subreddit_names_to_follow = [ 'getmotivated', 'adviceanimals',
                             'europe']

with sqlite3.connect('reddit.db') as conn:
    res = conn.execute('''select distinct *
    from subreddits order by display_name''').fetchall()
    for i in res:
        print(i)
    print(len(res))