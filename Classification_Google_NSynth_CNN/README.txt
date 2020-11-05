1. Within the "Code" directory, you may find 4 python programs.

q1 - Initial network for Instrument families
q2 - Advanced network for Instrument families
q3 - Initial network for Instrument Sources
q4 - Advanced network for Instrument Sources

To run any of these programs, you may just run it with python3 with no parameters required to pass to the program.

2. You may find 4 directories inside the "Code" directory. Each of these directories have all images, plots, confusion matricies, learning rates and WEIGHT files corresponding to each program.

Q1 - All files related to the Initial network's for Instrument families
Q2 - All files related to the advanced network's for Instrument families
Q3 - All files related to the Initial network's for Instrument Sources
Q4 - All files related to the advanced network's for Instrument Sources

3. To use the weight files that are already trained, just copy paste the weight file next to the q<x>.py file (q1.py for example). If a valid weight file is found, the program will automatically pick the weight file and just classify without running any training epocs. Otherwise, it will run the epocs and generate all of these images and weight files within the corresponding directory.
