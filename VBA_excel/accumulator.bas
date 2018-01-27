
Sub accumulate()
Dim extwbk As Workbook, twb As Workbook
Set twb = ThisWorkbook

For code = 1 To 49 Step 1
    If code = 1 Then
        Workbooks.Open "path_to_input_table\input_table_1"
    ElseIf code = 2 Then
        Workbooks.Open "path_to_input_table\input_table_2"
    ElseIf code = 3 Then
        Workbooks.Open "path_to_input_table\input_table_3"
    ElseIf code = 4 Then
        Workbooks.Open "path_to_input_table\input_table_4"
    ElseIf code = 5 Then
        Workbooks.Open "path_to_input_table\input_table_5"
    ElseIf code = 6 Then
        Workbooks.Open "path_to_input_table\input_table_6"
    ElseIf code = 7 Then
        Workbooks.Open "path_to_input_table\input_table_7"
    ElseIf code = 8 Then
        Workbooks.Open "path_to_input_table\input_table_8"
    ElseIf code = 9 Then
        Workbooks.Open "path_to_input_table\input_table_9"
    ElseIf code = 10 Then
        Workbooks.Open "path_to_input_table\input_table_10"
    ElseIf code = 11 Then
        Workbooks.Open "path_to_input_table\input_table_11"
    ElseIf code = 12 Then
        Workbooks.Open "path_to_input_table\input_table_12"
    ElseIf code = 13 Then
        Workbooks.Open "path_to_input_table\input_table_13"
    ElseIf code = 14 Then
        Workbooks.Open "path_to_input_table\input_table_14"
    ElseIf code = 15 Then
        Workbooks.Open "path_to_input_table\input_table_15"
    ElseIf code = 16 Then
        Workbooks.Open "path_to_input_table\input_table_16"
    ElseIf code = 17 Then
        Workbooks.Open "path_to_input_table\input_table_17"
    ElseIf code = 18 Then
        Workbooks.Open "path_to_input_table\input_table_18"
    ElseIf code = 19 Then
        Workbooks.Open "path_to_input_table\input_table_19"
    ElseIf code = 20 Then
        Workbooks.Open "path_to_input_table\input_table_20"
    ElseIf code = 21 Then
        Workbooks.Open "path_to_input_table\input_table_21"
    ElseIf code = 22 Then
        Workbooks.Open "path_to_input_table\input_table_22"
    ElseIf code = 23 Then
        Workbooks.Open "path_to_input_table\input_table_23"
    ElseIf code = 24 Then
        Workbooks.Open "path_to_input_table\input_table_24"
    ElseIf code = 25 Then
        Workbooks.Open "path_to_input_table\input_table_25"
    ElseIf code = 26 Then
        Workbooks.Open "path_to_input_table\input_table_26"
    ElseIf code = 27 Then
        Workbooks.Open "path_to_input_table\input_table_27"
    ElseIf code = 28 Then
        Workbooks.Open "path_to_input_table\input_table_28"
    ElseIf code = 29 Then
        Workbooks.Open "path_to_input_table\input_table_29"
    ElseIf code = 30 Then
        Workbooks.Open "path_to_input_table\input_table_30"
    ElseIf code = 31 Then
        Workbooks.Open "path_to_input_table\input_table_31"
    ElseIf code = 32 Then
        Workbooks.Open "path_to_input_table\input_table_32"
    ElseIf code = 33 Then
        Workbooks.Open "path_to_input_table\input_table_33"
    ElseIf code = 34 Then
        Workbooks.Open "path_to_input_table\input_table_34"
    ElseIf code = 35 Then
        Workbooks.Open "path_to_input_table\input_table_35"
    ElseIf code = 36 Then
        Workbooks.Open "path_to_input_table\input_table_36"
    ElseIf code = 37 Then
        Workbooks.Open "path_to_input_table\input_table_37"
    ElseIf code = 38 Then
        Workbooks.Open "path_to_input_table\input_table_38"
    ElseIf code = 39 Then
        Workbooks.Open "path_to_input_table\input_table_39"
    ElseIf code = 40 Then
        Workbooks.Open "path_to_input_table\input_table_40"
    ElseIf code = 41 Then
        Workbooks.Open "path_to_input_table\input_table_41"
    ElseIf code = 42 Then
        Workbooks.Open "path_to_input_table\input_table_42"
    ElseIf code = 43 Then
        Workbooks.Open "path_to_input_table\input_table_43"
    ElseIf code = 44 Then
        Workbooks.Open "path_to_input_table\input_table_44"
    ElseIf code = 45 Then
        Workbooks.Open "path_to_input_table\input_table_45"
    ElseIf code = 46 Then
        Workbooks.Open "path_to_input_table\input_table_46"
    ElseIf code = 47 Then
        Workbooks.Open "path_to_input_table\input_table_47"
    ElseIf code = 48 Then
        Workbooks.Open "path_to_input_table\input_table_48"
    ElseIf code = 49 Then
        Workbooks.Open "path_to_input_table\input_table_49"
    End If
    
	'area of each table to select
    Sheets(1).Range("A1:B24").Select
    Selection.Copy
    
    'paste content of each selection
    ThisWorkbook.Activate
    Cells(1, 2 * code - 1).Select
    Selection.PasteSpecial Paste:=xlPasteAll, Operation:=xlNone, SkipBlanks:= _
        False
    Application.CutCopyMode = False
    
Next code
End Sub


