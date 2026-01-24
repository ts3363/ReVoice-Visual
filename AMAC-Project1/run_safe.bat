@echo off
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

echo ===== AMAC SAFE TRAIN STARTED ===== >> overnight_log.txt

:loop
python -m training.train_video >> overnight_log.txt 2>&1
echo Restarting after crash at %date% %time% >> overnight_log.txt
timeout /t 10
goto loop
