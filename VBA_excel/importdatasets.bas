Option Compare Database

Sub importdatasets()

'This sub import multiple Access Queries at the same time in different excel files Excel Binary Workbook files. 

For code = 1 To 1 Step 1

    tableToImport = "path_to_table_to_import\table_to_import.xlsx"
    DoCmd.TransferSpreadsheet acImport, acSpreadsheetTypeExcel12, "AccessTable_" & Str(code), tableToImport, True

Next code

End Sub

