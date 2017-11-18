import sqlite3

with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select *
    from comments a join comments b on a.c_id = b.parent_id
    join posts c on a.p_id = c.p_id order by b.submitted_timestamp desc''').fetchall()
        print(len(res))