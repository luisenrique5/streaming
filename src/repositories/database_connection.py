from sqlalchemy import MetaData, Table, select

class DatabaseConnection:
    def __init__(self, engine):

        self.__engine__ = engine
        self.__metadata__ = MetaData()

    def get_table(self, table_name: str):
        return Table(table_name, self.__metadata__, autoload_with=self.__engine__)

    def get_all_especific_table(self, table_name):
        especific_table = self.get_table(table_name)
        query = select(especific_table)
        return query
      
    def execute_query(self, query, json=False):
        try:
            with self.__engine__.connect() as connection:
                result = connection.execute(query)
                if json:
                    return result.scalar()
                else:
                    return result.fetchall()
        except Exception as e:
            raise ValueError(f'Error on Execute Query: {str(e)}')    