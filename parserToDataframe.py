
from openpyxl import load_workbook
from pandas import DataFrame as df



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


df_index = 'time'

sheet_ranges_example = load_workbook(filename = 'dataset_example.xlsx')['Sheet1']
parsedInDataframe = parserToDataframe(sheet_ranges_example).set_index(df_index)

