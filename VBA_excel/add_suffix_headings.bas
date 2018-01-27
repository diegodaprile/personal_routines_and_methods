
Sub add_suffix_to_variables()

'The macro substitutes the name of the variables in the first row with their initial content plus a suffix, in this case _c_<country_code>, where <country_code> is the number indicating the dataset's country.
'in the case of 49 datasets, we will have:

For code = 1 To 49 Step 1
    Workbooks.Open "path_to_dataset\input_dataset_" & Str(code)
    
	Sheets(1).Range("A1", Range("A1").End(xlToRight)).Select
    Selection.Copy
        
    Workbooks.Add
    ActiveCell.Select
    Selection.PasteSpecial Paste:=xlPasteAll, Operation:=xlNone, SkipBlanks:=False
    
    ActiveCell.Range("A1").Select
    num = WorksheetFunction.CountA(Range("B1:EN1"))
    
	'here the substitution:
    For i = 1 To num Step 1
        Cells(1, i + 1) = Cells(1, i + 1) & "_" & "c_" & Str(code)
    Next i
	
    'change the output directory 
    ChDir "path_to_output_directory\transposed_for_time_series_analysis" 
    
	'save the result
    ActiveWorkbook.SaveAs Filename:= _
        "path_to_output_directory\transposed_for_time_series_analysis\output_dataset_"  & Str(code), FileFormat:=xlOpenXMLWorkbook, CreateBackup:=False
    
	'close the workbook, do not save, as we already did it	
    Workbooks("output_dataset_" & Str(code)).Close savechanges:=False
	
    'close the input dataset, without saving it
    Workbooks("input_dataset_" & Str(code) & ".xlsx").Close savechanges:=False
Next code

End Sub