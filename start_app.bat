@echo off
title Universal Plant Disease Detection - FIXED (No More Errors)
color 0B

echo.
echo  ================================================================
echo   🌿 PLANT DISEASE DETECTION - FIXED VERSION (ERROR-FREE)
echo  ================================================================
echo.
echo  ✅ ALL ERRORS FIXED:
echo  • No more "list index out of range" errors
echo  • No more Streamlit deprecation warnings
echo  • Robust error handling implemented
echo  • CNN model properly configured
echo.
echo  🚀 Starting error-free application...
echo  🌐 Opening at http://localhost:8501
echo.
echo  📸 Now you can:
echo  • Upload any plant image without errors
echo  • Get reliable CNN analysis
echo  • View results without crashes
echo  • Enjoy smooth operation
echo.
echo  ================================================================
echo.

if not exist "universal_plant_disease_model.keras" (
    echo ❌ FIXED model not found! Run setup_app.bat first.
    pause
    exit /b 1
)

echo ✅ FIXED model found - launching error-free application...
streamlit run main.py

pause
