
Sub transpose_all_datasets()

'This sub transposes multiple tables in different excel files. The final output, output_dataset_<country_code>, is a series of newly created excel files with the transposed table inside.
'The following macro opens up a series of spreadsheets from different input files, named "input_dataset_<country_code>" where <country_code> is the number indicating the dataset's country 
'in the case of 49 datasets, we will have:

For code = 1 To 49 Step 1

	'open the datasets to transpose
    Workbooks.Open "path_to_dataset\input_dataset_" & Str(code)
	'range to select and copy
	Sheets(1).Range("A1", Range("A1").End(xlToRight)).Select
    Selection.Copy
    
	'creating new workbook
    Workbooks.Add
    ActiveCell.Select
    'paste the content, with Transpose option equal to True
	Selection.PasteSpecial Paste:=xlPasteAll, Operation:=xlNone, SkipBlanks:= _
        False, Transpose:=True
    Application.CutCopyMode = False
    
        ActiveCell.Offset(6, 0).Range("A1").Activate
        ActiveCell.Offset(-6, 0).Range( _
            "1:1,2:2,3:3,4:4,6:6,7:7,8:8,46:46,47:47,48:48,49:49,50:50,51:51,52:52,53:53,54:54,55:55,56:56" _
            ).Select
        Selection.Delete Shift:=xlUp
        ActiveCell.Range("A1").Select
        Selection.AutoFilter
        ActiveWorkbook.Worksheets("Sheet1").AutoFilter.Sort.SortFields.Clear
        ActiveWorkbook.Worksheets("Sheet1").AutoFilter.Sort.SortFields.Add Key:= _
            ActiveCell.Range("A1:A41"), SortOn:=xlSortOnValues, Order:= _
            xlDescending, DataOption:=xlSortNormal
        With ActiveWorkbook.Worksheets("Sheet1").AutoFilter.Sort
            .Header = xlYes
            .MatchCase = False
            .Orientation = xlTopToBottom
            .SortMethod = xlPinYin
            .Apply
        End With
        
    'change the output directory    
    ChDir "path_to_output_directory\transposed_for_time_series_analysis"
    
	'save the result
    ActiveWorkbook.SaveAs Filename:= _
        "path_to_output_directory\transposed_for_time_series_analysis\output_dataset_" & Str(code), FileFormat:=xlOpenXMLWorkbook, CreateBackup:=False
    
	'close the workbook, do not save, as we already did it  
    Workbooks("output_dataset_" & Str(code)).Close savechanges:=False
    
	'close the input dataset, without saving it
    Workbooks("input_dataset" & Str(code) & ".xlsb").Close savechanges:=False
Next code

End Sub