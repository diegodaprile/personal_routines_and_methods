Option Base 1
Option Compare Database


Sub looping_queries()

Dim strSQL As String

Dim dbs As Database
Set dbs = CurrentDb()
Dim rs As Recordset
Dim qdf As QueryDef

With dbs

For code = 1 To 49 Step 1

strSQL = "SELECT [Table].* " & _
         "FROM [Table] " & _
         "WHERE [Table].variable = " & Str(code) & ";"

query_n = "query_name_number_" & Str(code)
Set qdf = .CreateQueryDef(query_n, strSQL)
DoCmd.OpenQuery query_n
'.QueryDefs.Delete "test"

Next code

End With

dbs.Close
qdf.Close

End Sub





