Get-ChildItem "." -Filter *.csv | ForEach-Object {
    ECHO $_.FullName >> "output.txt"
    python ..\..\csv_hanlder.py $_ *>> "output.txt"
    ECHO "" >> "output.txt"
}