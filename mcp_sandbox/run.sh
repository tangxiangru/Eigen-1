cd ./api_proxy

srun -w SH-IDCA1404-10-140-54-77 -p p-cpu-new -J hle_server --kill-on-bad-exit=1 --async -o job/run_%j.log \
    python api_server.py

sleep 5

cd ..//MCP

srun -w SH-IDCA1404-10-140-54-77 -p p-cpu-new -J hle_server --kill-on-bad-exit=1 --async -o job/run_%j.log \
    bash deploy_server.sh


rm batchscript-2025*