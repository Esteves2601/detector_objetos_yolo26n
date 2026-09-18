Dim fso, pasta, shell
Set fso = CreateObject("Scripting.FileSystemObject")
pasta = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
' pyw = Python sem console (sem janela preta)
shell.Run "pyw """ & pasta & "\app_gui.py""", 0, False
Set shell = Nothing
Set fso = Nothing
