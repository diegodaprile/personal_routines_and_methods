Attribute VB_Name = "deleting_rows"
Option Compare Database

Sub deletesomerows()

Dim strSQL As String

Dim dbs As Database
Set dbs = CurrentDb()
Dim rs As Recordset
Dim qdf As QueryDef

With dbs

For code = 1 To 2 Step 1

strSQL = "DELETE * FROM [levels]" & _
         "WHERE [levels].Indicator =" & Str(WDI_Population_growth_(annual_%)) & ";"
         
query_n = "filtered_" & Str(code)
Set qdf = .CreateQueryDef(query_n, strSQL)
DoCmd.OpenQuery query_n
'.QueryDefs.Delete "test"

Next code

End With

dbs.Close
qdf.Close


End Sub

