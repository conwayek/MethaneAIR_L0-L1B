# MethaneAIR_L0-L1B

Listed here are the L0-L1B processing algorithms. 
run_L1B.py calls the main algorithm, MethaneAIR_L1B.py. 


Generally, for MethaneAIR, a list of files are initially created, on a flight-by-flight basis and a batch processing script is ran. 
These are the L0 files and are .seq files. The list is usually named 'RF0X_O2_seq.txt'
An iteration number (file number) is stored in iteration.txt, created by the batch submission script (bash). 
This tells the code which file to process in the list, i.e. which iteration. 
There are likely a million other ways to do this, this is just how I wrote it for MethaneAIR.

For MethaneSAT, it is different, individual files are passed as an input. i.e. as they are downlinked, converted from raw-->L0, they can be processed. 

Darkfiles are initially hardcoded for the first two research flights - as there are only two per flight. 
This will be changing moving forward as we are collecting more - note the use of 'darkfinder.py', which will be used in the next flights. 
The initial thinking is that a darkfile list will be created, and the most recent (closest) will be loaded to process a respective granule.

I have hardcoded a few variables - the type of computer, the flight number etc. 
The directories are pretty standard:
1. The L0 file directories. 
2. The root_data/calibration file directory.
3. Absolute akaze directories. 
4. Relative akaze directories.
5. L1B native directories.
6. L1B aggregated directories.


