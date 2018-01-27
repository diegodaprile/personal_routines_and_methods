Option Base 1
Option Compare Database

Sub exportingstuff()

'This sub export multiple Access Queries at the same time in different excel files Excel Binary Workbook files. 
'The final output is exported_table_<country_code>, where <country_code> is the number indicating the dataset's country.
'in the case of 49 previously executed Access queries, we will have:

For code = 1 To 49 Step 1

    strPath = "C:\Users\diego\Dropbox\BDA_teamgroup\dataset_used\exported_table_" & Str(code)
    DoCmd.TransferSpreadsheet acExport, acSpreadsheetTypeExcel12, "accessQueryToExport_" & Str(code), strPath

Next code

End Sub


