# MethaneAIR_L0-L1B

Listed here are the L0-L1B processing algorithms. 
run_L1B.py calls the main algorithm, MethaneAIR_L1B.py. 


Generally, for MethaneAIR, a list of files are initially created, on a flight-by-flight basis and a batch processing script is ran. 
An iteration number (file number) is stored in iteration.txt, which tells the code which file to process in the list. 
There are a million other ways to do this, this is just how I wrote it. 
I would prefer not to alter it at the present time, with MethaneAIR flights coming up soon - after these are processed, sure. 
For MethaneSAT, it is different, individual files are passed as an input. 


Darkfiles are initially hardcoded for the first two research flights - as there are only two per flight. 
This will be changing moving forward as we are collecting more. 
The initial thinking is that a darkfile list will be created, and the most recent (closest) will be loaded to process a respective granule.


