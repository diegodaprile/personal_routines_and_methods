Attribute VB_Name = "LoopingThroughNations"
Option Base 1
Option Compare Database


Sub filtering_nations()

Dim strSQL As String

Dim dbs As Database
Set dbs = CurrentDb()
Dim rs As Recordset
Dim qdf As QueryDef

With dbs

For code = 1 To 49 Step 1

strSQL = "SELECT [Levels].* " & _
         "FROM [Levels] " & _
         "WHERE [Levels].country_code = " & Str(code) & ";"

query_n = "nation_" & Str(code)
Set qdf = .CreateQueryDef(query_n, strSQL)
DoCmd.OpenQuery query_n
'.QueryDefs.Delete "test"

Next code

End With

dbs.Close
qdf.Close

End Sub



