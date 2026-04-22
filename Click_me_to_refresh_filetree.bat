@echo off
setlocal enabledelayedexpansion

:: 1. 进入当前脚本所在目录
cd /d "%~dp0"

echo ========================================
echo 正在处理项目：FLIPO_FLIP
echo ========================================

:: 目标 1: 生成 folder_structure.txt (彻底排除 .venv 文件夹)
:: 使用 PowerShell 递归生成目录树，并在搜索时直接跳过名为 .venv 的目录
echo [1/2] 正在更新文件夹结构（已彻底过滤 .venv）...
powershell -Command "$exclude = @('.venv'); function Get-Tree($path, $indent='') { Get-ChildItem -Path $path | Where-Object { $exclude -notcontains $_.Name } | ForEach-Object { $item = $_; $item.Name; if ($item.PSIsContainer) { Get-Tree $item.FullName ($indent + '    ') | ForEach-Object { $indent + '    ' + $_ } } } } Get-Tree ." > folder_structure.txt
echo [OK] folder_structure.txt 已更新！

:: 目标 2: 在各自的文件夹下生成 requirements.txt
echo [2/2] 正在导出各环境的依赖...

:: 遍历当前目录下所有名为 .venv 的文件夹
for /r /d %%D in (*.venv) do (
    set "VENV_PATH=%%D"
    
    :: 获取 .venv 所在的父文件夹路径（例如 D:\...\FLIPO_FLIP\Simulation）
    for %%P in ("!VENV_PATH!\..") do set "PARENT_DIR=%%~fP"
    
    :: 检查该环境下是否存在 pip.exe
    if exist "!VENV_PATH!\Scripts\pip.exe" (
        echo 正在导出环境依赖至: !PARENT_DIR!\requirements.txt
        "!VENV_PATH!\Scripts\pip.exe" freeze > "!PARENT_DIR!\requirements.txt"
        echo [OK] 已完成！
    )
)

echo ========================================
echo 任务完成！requirements 已存入各自子目录。bz！
echo ========================================
pause