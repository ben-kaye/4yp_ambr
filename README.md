# 4YP: Automated microBioreactor (AmBR)

The source code for running the reactor experiments is included in /BR-OS/.

Running an experiment is as follows. First update the directories in the ./BR-OS/data/settings.json. 
1. python32 > ./BR-OS/scripts/run_scan.py
2. python64 > ./BR-OS/scripts/get_wells.py with a scan chosen
3.          > ./BR-OS/scripts/run_processor.py for online processing

Alternatively run_offline_processor.py to reprocess data. Or run_controller.py to include control outputs. 
