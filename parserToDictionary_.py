

from openpyxl import load_workbook


def parserToDictionary(sheet):
    columns = []
    for row in sheet.rows:
        if columns:
            [columns[i].append(ref.value) for i, ref in enumerate(row)]
        else:
            columns = [[ref.value] for ref in row]
    result = {x[0] : x[1:] for x in columns}
    return result




sheet_ranges_example = load_workbook(filename = 'dataset_example.xlsx')['Sheet1']
newlyCreatedDictionary = parserToDictionary(sheet_ranges_example)


