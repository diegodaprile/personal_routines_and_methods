#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:29:40 2017

@author: DiegoCarlo
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
Diego
This is a temporary script file.
"""

from openpyxl import load_workbook
from pandas import DataFrame as df

def parserToDictionary(sheet):
    columns = []
    for row in sheet.rows:
        if columns:
                [columns[i].append(ref.value) for i, ref in enumerate(row)]
        else:
            columns = [[ref.value] for ref in row]
    result = {x[0] : x[1:] for x in columns}
    return result


def parserToDataframe(sheet):
    columns = []
    for row in sheet.rows:
        if columns:
            for i, ref in enumerate(row):
                columns[i].append(ref.value)
        else:
            columns = [[ref.value] for ref in row]
    storedInDictionary = {c[0] : c[1:] for c in columns}
    result = df(storedInDictionary)
    return result



sheet_ranges_example = load_workbook(filename = 'dataset_example_1.xlsx')['Sheet1']

newlyCreatedDictionary = parserToDictionary(sheet_ranges_example)
parsedInDataframe = parserToDataframe(sheet_ranges_example).set_index('time')

