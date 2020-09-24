# -*- conding:utf-8 -*-
import datetime
import cx_Oracle
import os
import uuid
class oracleOperation():
    def openOracleConn(self):

        # highway=cx_Oracle.connect('api_work','apiwork@1234',xnj)
        highway=cx_Oracle.connect('hnwsl','lsd','10.232.218.25:1521/hnnyj')
        #获取cursor指针
        # cursor=highway.cursor()
        return highway

    def select(self,connection):
        cursor = connection.cursor()
        # 数据库操作
        # 1.查询
        # sql = 'select * from AUTH_USERS'
        sql = "select * from C_TR2P_VALUES where param_id=\'PA430000230*\' and rd_id1=\'R43000230\'"
        result = cursor.execute(sql)
        print('type of result', type(result))  # 获取使用cursor对象的execute的方法返回的对象的类型
        print('result:', result)  # 获取使用cursor对象的execute的方法返回的对象
        print("Number of rows returned: %d" % cursor.rowcount)
        rows = cursor.fetchall()  # 得到所有数据集

        print("rows:", rows)  # fetchall（）方法得到的到底是设么类型的对象

        for i in rows:
            print('序号：', i[0])
            print('水果种类：', i[1])
            print('水果库存量：', i[2])
        cursor.close()





        #1.插入操作

    def select_C_TR2P_VALUES(self, connection, selectParam={}):
        #select只支持单条语句查询
        cursor = connection.cursor()
        # M=[(11,'sa','sa'),]
        sql = "select event_id from C_TR2P_VALUES where table_id=:table_id and edition_id=:edition_id and " \
              "source_id=:source_id and param_id=:param_id and timeflag_id=:timeflag_id and data_datetime=:data_datetime and rd_id1=:rd_id1 and " \
              "rd_id2=:rd_id2 and param_unit=:param_unit and data_type=:data_type"
        # sql = "select event_id from C_TR2P_VALUES where rd_id1=:rd_id1 and rd_id2=:rd_id2"
        rows = []
        if (len(selectParam) == 0):
            print("查询的数据行的参数不能为空！")
            return rows
        else:
            result = cursor.execute(sql,selectParam)
            rows = cursor.fetchall()  # 得到所有数据集

            print("Insert result:", rows)
            # count=cursor.execute("SELECT COUNT(*) FROM python_modules")
            # print("count of python_modules:",count)
        cursor.close()
        return rows

    def update_C_TR2P_VALUES(self, connection, updateParam=[]):
        cursor = connection.cursor()
        # M=[(11,'sa','sa'),]
        sql = "update C_TR2P_VALUES set param_value=:param_value,last_modified=:last_modified where table_id=:table_id and edition_id=:edition_id and " \
              "source_id=:source_id and param_id=:param_id and timeflag_id=:timeflag_id and data_datetime=:data_datetime and rd_id1=:rd_id1 and " \
              "rd_id2=:rd_id2 and param_unit=:param_unit and data_type=:data_type"
        if (len(updateParam) == 0):
            print("插入的数据行的参数不能为空！")
        else:
            # result = cursor.execute(sql, updateParam) #单条语句
            cursor.prepare(sql)
            result = cursor.executemany(None, updateParam)  #批量语句
            print("Insert result:", result)

        connection.commit()
        cursor.close()

    def insert_C_TR2P_VALUES(self,connection,insertParam=[]):
        cursor=connection.cursor()
        # M=[(11,'sa','sa'),]
        sql="insert into C_TR2P_VALUES (event_id,table_id,edition_id,source_id,param_id,timeflag_id,data_datetime,rd_id1,rd_id2,param_value,param_unit,input_datetime,last_modified,data_type) \
        values (:event_id,:table_id,:edition_id,:source_id,:param_id,:timeflag_id,:data_datetime,:rd_id1,:rd_id2,:param_value,:param_unit,:input_datetime,:last_modified,:data_type)"
        if(len(insertParam)==0):
            print("插入的数据行的参数不能为空！")
        else:
            cursor.prepare(sql)
            result=cursor.executemany(None,insertParam)

            print("Insert result:",result)
            # count=cursor.execute("SELECT COUNT(*) FROM python_modules")
            # print("count of python_modules:",count)

        connection.commit()
        cursor.close()

    def insert2(self,connection,insertParam=[]):
        cursor=connection.cursor()
        # M=[(11,'sa','sa'),]
        sql="insert into AUTH_USERS (USERNAME,PASSWORD,ENABLED) values (:id,:kinds,:numbers)"
        if(len(insertParam)==0):
            print("插入的数据行的参数不能为空！")
        else:
            cursor.prepare(sql)
            result=cursor.executemany(None,insertParam)

            print("Insert result:",result)
            # count=cursor.execute("SELECT COUNT(*) FROM python_modules")
            # print("count of python_modules:",count)

        connection.commit()
        cursor.close()
if __name__=='__main__':
    db = oracleOperation()
    connection=db.openOracleConn()

    start = '2020/7/11'
    end = '2020/8/19'
    value = '539'
    date_now = datetime.datetime.strptime(start, '%Y/%m/%d')

    param_select = {
        # 'event_id': uuid.uuid4().hex,
        'table_id': 'TA43030001',
        'edition_id': 'DEFAULT',
        'source_id': 'S430018',
        'param_id': 'PA430000230*',
        'timeflag_id': 'D',
        'data_datetime': date_now,
        'rd_id1': 'R43000230',
        'rd_id2': '430000',
        'param_unit': 'U430019',
        # 'input_datetime': date_now,
        # 'last_modified': date_now,
        'data_type': '1'
    }
    param_update = {
        # 'event_id': uuid.uuid4().hex,
        'table_id': 'TA43030001',
        'edition_id': 'DEFAULT',
        'source_id': 'S430018',
        'param_id': 'PA430000230*',
        'timeflag_id': 'D',
        'data_datetime': date_now,
        'param_value':'333333',
        'rd_id1': 'R43000230',
        'rd_id2': '430000',
        'param_unit': 'U430019',
        # 'input_datetime': date_now,
        'last_modified': datetime.datetime.now(),
        'data_type': '1'
    }
    param_0 = {
        'event_id': uuid.uuid4().hex,
        'table_id': 'TA43010001',
        'edition_id': 'DEFAULT',
        'source_id': 'S430008',
        'param_id': 'PA43000630',
        'timeflag_id': 'D',
        'data_datetime': date_now,
        'param_value': '333333',
        'rd_id1': 'R43001001',
        'rd_id2': '430000',
        'param_unit': 'U430002',
        'input_datetime': datetime.datetime.now(),
        'last_modified': datetime.datetime.now(),
        'data_type': '1'
    }
    #能运行的无条件查询语句
    # db.select(connection)


    while(date_now<datetime.datetime.strptime(end, '%Y/%m/%d')):
        insertParams = []
        print(date_now)
        param_insert = param_0
        param_insert['event_id'] = uuid.uuid4().hex
        param_insert['param_value'] = value
        param_insert['data_datetime'] = date_now
        insertParams.append(param_insert)
        db.insert_C_TR2P_VALUES(connection, insertParams)
        date_now = (date_now + datetime.timedelta(days=1))
        pass
    # db.select_C_TR2P_VALUES(connection,param_select)
    # db.update_C_TR2P_VALUES(connection,[param_update])