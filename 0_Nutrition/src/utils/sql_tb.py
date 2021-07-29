import pymysql

class MySQL:
    #####
    def __init__(self, IP_DNS, USER, PASSWORD, DB_NAME, PORT):
        self.IP_DNS = IP_DNS
        self.USER = USER
        self.PASSWORD = PASSWORD
        self.DB_NAME = DB_NAME
        self.PORT = PORT
        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.DB_NAME

    #####
    def connect(self):
        # Open database connection
        self.db = pymysql.connect(host = self.IP_DNS,
                                  user = self.USER,
                                  password = self.PASSWORD,
                                  database = self.DB_NAME,
                                  port = self.PORT)
        
        # Create a cursor object using cursor method
        self.cursor = self.db.cursor()
        print("Connected to MySQL server [" + self.DB_NAME + "]")
        return self.db

    #####
    def close(self):
        # Disconnect from server
        self.db.close()
        print("Close connection with MySQL server [" + self.DB_NAME + "]")

    #####
    def execute_interactive_sql(self, sql, delete = False):
        ''' NOT A SELECT COMMAND '''

        result = 0
        try :
            # Execute SQL command
            self.cursor.execute(sql)
            self.db.commit()
            print("Executed \n\n" + str(sql) + "\n\n succesfully")
            result = 1

        except Exception as error:
            print(error)
            # Rollback in case there is an error
            self.db.rollback()

        return result

    #####
    def execute_get_sql(self, sql):
        ''' SELECT command '''

        results = None
        print("Executing:\n", sql)
        try:
            # Execute SQL command
            self.cursor.execute(sql)

            # Fetch all the rows in a list of lists and save it in results
            results = self.cursor.fetchall()

        except Exception as error:
            print(error)
            print("Error: unable to fetch the data")

        return results      # list of lists

    #####
    def insert_into_predictions(self, to_insert):
        sql = """INSERT INTO predictions (model, RIDAGEYR, BPXDI1, BPXSY1, BMXWT, BMXWAIST, LBXTC, LBXSGL, MEANCHOL, MEANTFAT, MEANSFAT, MEANSUGR, MEANFIBE, MEANTVB6, FEMALE, MALE, prediction) VALUES ('""" + to_insert[0] + """', '""" + to_insert[1] + """', '""" + to_insert[2] + """', '""" + to_insert[3] + """', '""" + to_insert[4] + """', '""" + to_insert[5] + """', '""" + to_insert[6] + """', '""" + to_insert[7] + """', '""" + to_insert[8] + """', '""" + to_insert[9] + """', '""" + to_insert[10] + """', '""" + to_insert[11] + """', '""" + to_insert[12] + """', '""" + to_insert[13] + """', '""" + to_insert[14] + """', '""" + to_insert[15] + """', '""" + to_insert[16] + """')"""

        return sql