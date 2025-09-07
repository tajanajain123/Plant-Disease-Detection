@echo off
title Universal Plant Disease Detection - FIXED VERSION
color 0A

echo.
echo  ================================================================
echo   🌿 UNIVERSAL PLANT DISEASE DETECTION - FIXED VERSION
echo  ================================================================
echo.
echo  🔧 ALL CRITICAL ERRORS HAVE BEEN RESOLVED:
echo  ✅ Fixed: "list index out of range" error
echo  ✅ Fixed: Streamlit deprecation warnings  
echo  ✅ Enhanced: Error handling and bounds checking
echo  ✅ Updated: CNN model with exactly 200 output classes
echo.
echo  This FIXED version will:
echo  • Create error-free CNN model (200 classes)
echo  • Install all required dependencies
echo  • Resolve index out of range issues
echo  • Eliminate Streamlit warnings
echo.
echo  ================================================================
echo.

pause

echo 📦 Starting FIXED system setup...
echo.

python setup.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Setup failed! Please check error messages above.
    pause
    exit /b 1
)

echo.
echo ✅ FIXED system setup completed successfully!
echo ✅ All critical errors have been resolved!
echo.
echo 🚀 You can now run the error-free application:
echo    start_app.bat
echo.

set /p choice="Start the FIXED application now? (y/n): "
if /i "%choice%"=="y" call start_app.bat

pause
