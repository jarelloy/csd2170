::%%i 	- file
::%~dp0 - current dir
::%%~ni - file without extension
for /r %%i in (*.frag, *.vert, *.comp) do "%~dp0glslc.exe" "%%i" -o "%~dp0%%~ni.spv"
pause