# Astro Pi Competition - UTCSpace

I participated in the Astro Pi competition ran by the Raspberry Pi Foundation in collaboration with the European Space Agency, where I was tasked to calculate the linear velocity of the International Space Station using sensors and camera modules connected to a Raspberry Pi onboard the space station. My program achieved flight status, meaning that it will be sent up and run on the ISS in April/May 2024.

My program mainly utilises the camera module to capture photos of Earth at regular intervals, and then uses the opencv-python library to apply Computer Vision techniques to them to detect significant features within the photos. By filtering and comparing these features across pairs of images, it was able to identify the same features on different photos, which meant that the distance between them could be calculated. Then, by combining the distance with the amount of time elapsed between captures resulted in the linear velocity.

The requirements for the program set by the organisers can be found [here](https://astro-pi.org/mission-space-lab/rulebook).

## How to run the program

1. Install [Thonny](https://thonny.org/).
2. Download `main.py` to a folder of your chosen location (it temporarily stores photos in the folder so it is ideal to use a new one)
3. Navigate to `Tools > Manage Packages` and search `astro-pi-replay` in the search bar. Install the top result which description should be "A CLI to replay historic data from previous ISS missions."
4. Once installed, make sure that The folder selected by thonny is the one containing `main.py`.
5. In the terminal, run the following command: `!"Astro-Pi-Replay" run "main.py"` ([multiple warning messages](https://imgur.com/a/mIttGmQ) may be outputted in the terminal upon first time of running. These do not affect the running of the program because Thonny fixes them itself).
6. After at least 40 seconds, there should be at least one image in the folder, otherwise the program has not started successfully.
7. The program will run for ten minutes, and will output the resulting linear velocity to a text file named `results.txt` at the end of the program.
8. You can add print statements to output the list of speeds or any other values as well.

Disclaimer: The program does not actually output the correct linear velocity, due to inaccuracies with OpenCV and the filtering. An exact and correct result can be obtained by running `skyfield.py`. (This can be ran by following the steps above, but replacing `main.py` with `skyfield.py` in each of the steps)