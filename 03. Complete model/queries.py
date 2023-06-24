import pandas as pd
from connection import sql_connection

def select_query(fields,tables,union, where,base="dbo."):
    cursor,con = sql_connection()
    list_fields = ", ".join(fields)
    query = "SELECT "+ list_fields +" FROM "+ base + tables[0] 
    if union:
        for i in range(len(union)):
            query += " "+ union[i][0]+" JOIN "+base+ tables[i+1]+ " ON " +base+ union[i][1]+"="+base+ tables[i+1] + "." + union[i][2]
    if where:
        query+= " WHERE "
        for i in range(len(where)):
            query+= where[i][0] + " " + where[i][1] + " " + where[i][2]
            if(where[i][3]):
                query += where[i][3]
    cursor.execute(query)
    result = pd.DataFrame(cursor.fetchall(),columns=fields)
    cursor.close()
    return result

def create_query(table,**name):
    cursor,con = sql_connection()
    query = "CREATE TABLE {}(".format(table)
    for key, value in name.items():
        query +="{} {},".format(key,value)
    query = query[:-1]
    query+=");"
    cursor.execute(query)

def insert_query(table,columns,values):
    cursor,con = sql_connection()
    query = """INSERT INTO {}""".format(table) 
    empty = ['?'] * len(values)
    empty = ", ".join(empty)
    columns = ", ".join(columns)
    query += """({}) VALUES ({});""".format(columns,empty)
    cursor.execute(query,values)
    con.commit()
    cursor.close()

def drop_query(table):
    cursor, con = sql_connection()
    query = "DROP TABLE {}".format(table)
    cursor.execute(query)
    