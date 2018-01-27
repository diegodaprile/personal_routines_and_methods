

Sub changingNamesLookup()

'The macro changes the content of the first row with their correspondent abbreviated names. This is done through a simple VLOOKUP. This is done many times, for all the dataset named dataset_to_modify_ in the script.

Dim i As Long, code As Long
Dim extwbk As Workbook, twb As Workbook

Set twb = ThisWorkbook
Set extwbk = Workbooks.Open("path_for_table_to_lookup\table_for_lookup.xlsx")
'the table from which to lookup is 
Set lookup_table = extwbk.Worksheets("Sheet1").Range("C:D")

For code = 1 To 49 Step 1

    Workbooks.Open "path_to_dataset\dataset_to_modify_" & Str(code)
    ActiveCell.Range("A1").Select
    num = WorksheetFunction.CountA(Range("B1", Range("B1").End(xlToRight)))
    
    With twb.Sheets("Sheet1")
        
        For i = 1 To num Step 1
            Cells(1, i + 1) = Application.VLookup(Cells(1, 1 + i).Value, lookup_table, 2, False)
        Next i
	'close and save the changes
    Workbooks("dataset_to_modify_" & Str(code) & ".xlsx").Close savechanges:=True
    
    End With

Next code
End Sub
