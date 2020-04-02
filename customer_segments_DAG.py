#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:01:15 2020

@author: admin
"""


from airflow import DAG
import datetime as dt
from airflow.operators.bash_operator import BashOperator
from airflow.utils.email import send_email


def notify_email(contextDict, **kwargs):
    """Send custom email alerts."""

    # email title.
    title = "Airflow alert: {task_name} Failed".format(**contextDict)
    
    # email contents
    body = """
    Hi Everyone, <br>
    <br>
    There's been an error in the {task_name} job.<br>
    <br>
    Forever yours,<br>
    Airflow bot <br>
    """.format(**contextDict)

    send_email('example@company.io', title, body) 
    
    
    
default_args = {
        'owner': 'ivana',
        'start_date': dt.datetime(2020, 4, 1),
        'depends_on_past': False,
        'email': ['example@company.io'],
        'email_on_failure': False,
        'retries': 0,
        'retry_delay': dt.timedelta(minutes = 2) }

dag = DAG(
        'assign_segments',
        default_args = default_args,
        description = 'Assign already created segments to new customers',
        # Continue to run DAG once per day at midnight
        schedule_interval = '0 0 * * *',
        catchup = False)

assign_segments = BashOperator(task_id='assign_segments', 
                                                      bash_command='jupyter nbconvert --execute --to html $AIRFLOW_HOME/dags/common/New_data_segments.ipynb --no-input',
                                                      on_failure_callback=notify_email,dag=dag)

assign_segments