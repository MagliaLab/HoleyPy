'''
    This script was modified from PySential(TM) with permission of the original authors
    All copyright belongs to their respective holders.
'''

import sqlite3
import os.path


class SQL_database:
    def __init__(self, db_path):
        self.db_path = db_path

    def _push_query(self, query=None, fetch=False, commit=False):
        # print(query)
        if self.db_path:
            conn = self._load_database()
            if not conn:
                return False
            try:
                c = conn.cursor()
                if type(query) == str:
                    result = c.execute(query)
                elif type(query) == list:
                    [c.execute(q) for q in query]
                if commit:
                    conn.commit()
                if fetch:
                    return [list(i) for i in result.fetchall()]
                return True
            except sqlite3.DatabaseError as e:
                print(e)
                return False
            finally:
                conn.close()
        return False

    def _load_database(self):
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute('SELECT name from sqlite_master where type= "table"')
            return conn
        except sqlite3.DatabaseError as e:
            print(e)
            return False

    def make_database(self) -> bool:
        """Makes a new database in db_path

        Returns
        -------
        bool
        """
        conn = self._load_database()
        if conn:
            conn.close()
            return True
        return False

    def get_tables(self) -> list:
        """Get all tables in db_path

        Returns
        -------
        list
        """
        query = 'SELECT name from sqlite_master where type= "table";'
        return [i[0] for i in self._push_query(query=query, fetch=True)]

    def add_table(self, table_name='', fields='') -> bool:
        """Add new table to db_path

        Parameters
        ----------
        table_name : str
            Name of new table
        fields : str
            Fields string

        Returns
        -------
        bool
        """
        query = "CREATE TABLE " + table_name + " (" + fields + ");"
        return self._push_query(query=query, fetch=False)

    def get_fields(self, table_name='') -> list:
        """Gets all fields from table_name

        Parameters
        ----------
        table_name : str
            Table name to search for fields

        Returns
        -------
        list
        """
        query = 'SELECT sql FROM sqlite_master WHERE tbl_name="' + table_name + '" AND type="table";'
        result = self._push_query(query=query, fetch=True)[0][0]
        # print(result)
        fields = str(result[result.find("(") + 1:result.find("))")]).split(',')
        # print(fields)
        field_list = [tuple(i.split(" "))[0] for i in fields]
        return field_list

    def add_field(self, table_name='', field_info='') -> bool:
        """Add new field to table

        Parameters
        ----------
        table_name : str
            Table name to insert new field
        field_info : str
            Field information

        Returns
        -------
        bool
        """
        query = 'ALTER TABLE ' + table_name + ' ADD ' + field_info + ' VARCHAR;'
        return self._push_query(query=query)

    def get_samples(self, table_name='', field_name='*', search_query='*') -> list:
        """Get samples of specified table_name and field_name

        Parameters
        ----------
        table_name : str
            Table name to insert new field
        field_name : str
            Name of field(s) to search
        search_query : str
            Search query, default '*'

        Returns
        -------
        list
        """
        if search_query != '*':
            query = 'SELECT * FROM ' + table_name + ' WHERE '
            for c, field in enumerate(field_name):
                query = query + '(' + field + ' LIKE "%' + search_query + '%")'
                if c != len(field_name) - 1:
                    query = query + ' OR '
            query = query + ';'
        else:
            query = 'SELECT * FROM ' + table_name + ';'
        return self._push_query(query=query, fetch=True)

    def add_sample(self, table_name='', fields=None, data=None) -> bool:
        """Add sample to specified table_name and field_name

        Parameters
        ----------
        table_name : str
            Table name to insert new field
        fields : list
            Name of field(s)
        data : list
            List of data to add

        Returns
        -------
        bool
        """
        fields_string = ", ".join(["'" + str(i) + "'" for i in fields])
        data_string = ""
        for c, i in enumerate(data):
            if c != 0:
                data_string += ", "
            if type(i) == str:
                data_string += "'" + i + "'"
            else:
                data_string += str(i)
        query = "INSERT INTO '" + table_name + "' (" + fields_string + ") VALUES (" + data_string + ");"
        return self._push_query(query=query, commit=True)

    def add_samples(self, table_name='', fields=None, data_list=None) -> bool:
        """Add sample to specified table_name and field_name

        Parameters
        ----------
        table_name : str
            Table name to insert new field
        fields : list
            Name of field(s)
        data : list
            List of data to add

        Returns
        -------
        bool
        """
        fields_string = ", ".join(["'" + str(i) + "'" for i in fields])
        query_data = ""
        for n, data in enumerate(data_list):
            data_string = ""
            for c, i in enumerate(data):
                if c != 0:
                    data_string += ", "
                if type(i) == str:
                    data_string += "'" + i + "'"
                else:
                    data_string += str(i)
            query_data += '(' + data_string + ')'
            if n != len(data_list) - 1:
                query_data += ','
        query = "INSERT INTO '" + table_name + "' (" + fields_string + ") VALUES " + query_data + ";"
        return self._push_query(query=query, commit=True)

    def drop_trace(self, table_name='', trace=None) -> bool:
        if trace:
            query = "DELETE FROM " + table_name + " WHERE Trace=" + str(trace) + ";"
            # print(query)
            return self._push_query(query=query, commit=True)
        else:
            return False