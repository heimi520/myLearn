# -*- coding: utf-8 -*-
"""
Connector tools make connecting more easy.

Example:

    First, add your oracle connection info in a `conf.py` into "dstools/" path, 
    including ORACLE_CONF:

    ```json
    ORACLE_CONF = {
        'server': 'your servername',
        'port': 1521,
        'db': 'your data base name',
        'user': 'username',
        'psw': 'password'
    }
    ```
    Then just use it add the path and import the module.

    ```python
    import os, sys
    sys.path.append('../meorient_ds/')
    from dstools import oracle_conn as oc
    import pandas as pd

    df = pd.read_sql('your_sql', oc())
    ```

@author: huangbaochen
@email: huangbaochen@meorient.com
"""

import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


import cx_Oracle
from conf import *
from sqlalchemy import create_engine, types
import pandas as pd
import os

#os.environ["NLS_LANG"] = ".AL32UTF8"

CREATE_TRIGGER = """
CREATE OR REPLACE TRIGGER {trigger_name}
BEFORE INSERT ON {table_name}
FOR EACH ROW
BEGIN
  select sysdate into :new.created_at from dual;
END
"""

UPDATE_TRIGGER = """
CREATE OR REPLACE TRIGGER {trigger_name}
BEFORE INSERT OR UPDATE ON {table_name}
FOR EACH ROW
BEGIN
  select sysdate into :new.updated_at from dual;
END
"""

## Oracle connector
def oracle_conn():
    """oracle connector
    Connector tools make connecting more easy.

    Parameters:
    None

    Return:
    A pandas connector.

    Example:

        First, add your oracle connection info in a `conf.py` into "dstools/" path, 
        including ORACLE_CONF:

        ```json
        ORACLE_CONF = {
            'server': 'your servername',
            'port': 1521,
            'db': 'your data base name',
            'user': 'username',
            'psw': 'password'
        }
        ```
        Then just use it add the path and import the module.

        ```python
        import os, sys
        sys.path.append('../meorient_ds/')
        from dstools import oracle_conn as oc
        import pandas as pd

        df = pd.read_sql('your_sql', oc())
        ```

    @author: huangbaochen
    @email: huangbaochen@meorient.com
    """
    tns = cx_Oracle.makedsn(ORACLE_CONF['server'], 
                        ORACLE_CONF['port'], 
                        ORACLE_CONF['db'])
    conn = cx_Oracle.connect(ORACLE_CONF['user'], 
                        ORACLE_CONF['pwd'], tns, 
                        encoding='utf-8')
    return conn

    ## Oracle connector


def oracle88_conn():
    """oracle connector
    Connector tools make connecting more easy.

    Parameters:
    None

    Return:
    A pandas connector.

    Example:

        First, add your oracle connection info in a `conf.py` into "dstools/" path, 
        including ORACLE_CONF:

        ```json
        ORACLE_CONF = {
            'server': 'your servername',
            'port': 1521,
            'db': 'your data base name',
            'user': 'username',
            'psw': 'password'
        }
        ```
        Then just use it add the path and import the module.

        ```python
        import os, sys
        sys.path.append('../meorient_ds/')
        from dstools import oracle_conn as oc
        import pandas as pd

        df = pd.read_sql('your_sql', oc())
        ```

    @author: huangbaochen
    @email: huangbaochen@meorient.com
    """
    tns = cx_Oracle.makedsn(ORACLE88_CONF['server'],
                            ORACLE88_CONF['port'],
                            ORACLE88_CONF['db'])
    conn = cx_Oracle.connect(ORACLE88_CONF['user'],
                             ORACLE88_CONF['pwd'], tns,
                             encoding='utf-8')
    return conn


def backup_conn():
    """oracle connector
    Connector tools make connecting more easy.

    Parameters:
    None

    Return:
    A pandas connector.

    Example:

        First, add your oracle connection info in a `conf.py` into "dstools/" path, 
        including ORACLE_CONF:

        ```json
        ORACLE_CONF = {
            'server': 'your servername',
            'port': 1521,
            'db': 'your data base name',
            'user': 'username',
            'psw': 'password'
        }
        ```
        Then just use it add the path and import the module.

        ```python
        import os, sys
        sys.path.append('../meorient_ds/')
        from dstools import oracle_conn as oc
        import pandas as pd

        df = pd.read_sql('your_sql', oc())
        ```

    @author: huangbaochen
    @email: huangbaochen@meorient.com
    """
    tns = cx_Oracle.makedsn(BACKUP_CONF['server'],
                            BACKUP_CONF['port'],
                            BACKUP_CONF['db'])
    conn = cx_Oracle.connect(BACKUP_CONF['user'],
                             BACKUP_CONF['pwd'], tns,
                             encoding='utf-8')
    return conn

def backup_conn():
    """oracle connector
    Connector tools make connecting more easy.

    Parameters:
    None

    Return:
    A pandas connector.

    Example:

        First, add your oracle connection info in a `conf.py` into "dstools/" path, 
        including ORACLE_CONF:

        ```json
        ORACLE_CONF = {
            'server': 'your servername',
            'port': 1521,
            'db': 'your data base name',
            'user': 'username',
            'psw': 'password'
        }
        ```
        Then just use it add the path and import the module.

        ```python
        import os, sys
        sys.path.append('../meorient_ds/')
        from dstools import oracle_conn as oc
        import pandas as pd

        df = pd.read_sql('your_sql', oc())
        ```

    @author: huangbaochen
    @email: huangbaochen@meorient.com
    """
    tns = cx_Oracle.makedsn(BACKUP_CONF['server'],
                            BACKUP_CONF['port'],
                            BACKUP_CONF['db'])
    conn = cx_Oracle.connect(BACKUP_CONF['user'],
                             BACKUP_CONF['pwd'], tns,
                             encoding='utf-8')
    return conn

def oraclebi_conn():
    """oracle connector
    Connector tools make connecting more easy.

    Parameters:
    None

    Return:
    A pandas connector.

    Example:

        First, add your oracle connection info in a `conf.py` into "dstools/" path, 
        including ORACLE_CONF:

        ```json
        ORACLE_CONF = {
            'server': 'your servername',
            'port': 1521,
            'db': 'your data base name',
            'user': 'username',
            'psw': 'password'
        }
        ```
        Then just use it add the path and import the module.

        ```python
        import os, sys
        sys.path.append('../meorient_ds/')
        from dstools import oracle_conn as oc
        import pandas as pd

        df = pd.read_sql('your_sql', oc())
        ```

    @author: huangbaochen
    @email: huangbaochen@meorient.com
    """
    tns = cx_Oracle.makedsn(ORACLEBI_CONF['server'],
                            ORACLEBI_CONF['port'],
                            ORACLEBI_CONF['db'])
    conn = cx_Oracle.connect(ORACLEBI_CONF['user'],
                             ORACLEBI_CONF['pwd'], tns,
                             encoding='utf-8')
    return conn

SHOW_SQL = """
SELECT 
column_name "Name", 
nullable "Null?",
concat(concat(concat(data_type,'('),data_length),')') "Type"
FROM user_tab_columns
WHERE table_name='{}'
"""

e_bi = create_engine("""oracle+cx_oracle://{user}:{pwd}@{server}:{port}/{db}""".format(**ORACLEBI_CONF))


## useful table handle tools
truncate_table = lambda table_name: e_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: e_bi.execute('DROP TABLE {}'.format(table_name))
def describe_table(table_name, e):
    """describe a table's schema
    """
    return pd.read_sql(SHOW_SQL.format(table_name), e)

describe_backup_table = lambda x: describe_table(x, backup_conn())
describe_bi_table = lambda x: describe_table(x, oraclebi_conn())

def update_trigger(trigger_name, table_name):
    e_bi.execute(UPDATE_TRIGGER.format(table_name=table_name, trigger_name=trigger_name))
    
def create_trigger(trigger_name, table_name):
    e_bi.execute(CREATE_TRIGGER.format(table_name=table_name, trigger_name=trigger_name))