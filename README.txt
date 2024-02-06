-----How to Install-----
This is for linux, idk how python packages work in windows
*Download the project, there's not really a "release" because it's a bunch of .py files, there's nothing to compile or build
*Open a terminal in the project folder
*Create a virtual environment using 'python3 -m venv .venv'
*Activate the virtual environment with 'source .venv/bin/activate'
*Using pip, install the following packages:
    numpy
    onnxruntime
    opencv-python

On the Vtube Studio side:
*go to .../VTube Studio/VTube Studio_Data/StreamingAssets/
*create a file named 'ip.txt'
*put the following in ip.txt:
# To listen for remote connections, change this to 0.0.0.0 or your actual IP on the desired interface.
ip=127.0.0.1

# This is the port the server will listen for tracking packets on.
port=11573

And you're done, it's installed
you can run it by running my facetracking.sh shell script which applies some settings I found useful for my camera (though I'm unsure how compatible they are with other setups), or you can just use 'python3 facetracker.py' after re-activating the virtual environment



-----Using the software-----
Generally you should be able to run it and just open the network webcam in VTS
I don't remember if there are other steps

but here are some things I've noticed that should make you have an easier time
*Running resolutions above 480p doesn't seem to improve tracking and causes a lot of latency with the webcam
*Running at higher resolutions might work if you use a lower framerate like 24fps, buy it's not worth it imo, your face gets downscaled to 224x224 before getting sent off to tracking
*Lighting largely doesn't matter from my testing (this was a goal of mine)
*Idk how well my automatic brightness works with darker skin, but if it's an issue you can edit the targetBrightness variable in facetracker.py
*30fps works great, but if you can somehow feed this more *I think* it can do more, based on frame times it should be able to do 45-50fps, but my webcam doesn't go that fast
*you can pass the --preview 1 flag to get a preview window, this shows you the webcam feed after having the brightness adjusted


-----Changes from Openseeface(in no particular order)-----

-Major restructure
-I've broken out major chunks of functionality into separate files to make them easier to deal with and read
    -I've tried to break functions down into multiple smaller parts when possible
    -I've taken a more object oriented approach in some places
    -I've removed features that didn't fit my needs
    -Removed functionlaity for tracking multiple faces
    -Removed options I didn't forsee myself using (this was intended to be just for me)
-Added more threading and multi-processing
    -The webcam (and some new image processing) are now on a separate process
    -Previews are now a separate process because I didn't want to deal with the related performance weirdness
    -Console messages are now handled via a helper thread, idk if it does any good, but it can't hurt
    -VTS communication is also handled via a helper thread
-Added image pre-processing
    -I'm now applying a gamma curve to the webcam output to make faces more visible
    -The gamma curve is calculated after face tracking is done, it uses a copy of the webcam frame and face location data so get the average brightness of the face
    -This has drasitcally improved low light performance
    -It also lets me run the webcam with no gain
    -Removed the separate library that handled the webcam, then broke that off into a separate library again
    -The stuff around the webcam is still simpler now, most of the existing cases didn't apply to my use case
-Revamped the way features are calculated and normalized
    -This was kind of my original intent, my eyes kept registering as closed when they were not
    -Removed features that didn't seem to be used by Vtube Studio
    -Added functionality to apply response curves to features
    -Removed the calibration period in favor of a system where the limits of features slowly decay towards a center point
    -Removed the average from the way featured are normalized
-Added a feature to prevent errant eye movements
    -Each eye has an average confidence and standard deviation calculated every frame
    -eye movements more than two standard deviations below average are severely restricted
    -The restriction is based on how far the confidence is from the cutoff point
-Added early exits to situations where subsequent steps will fail
-The main process skips frames when the webcam falls behind, once webcam latency is long enough to be late it will remain behind otherwise
-Added warnings to various states such as late webcam frames, longer than ideal frames, and early exits
-Changed which stats are shown on exit, added some stats that get tracked like webcam latency
-Added conditions to skip steps when they are unnecessary, such as not rotating the eye images for small angles


-----Credit-----
For documentation on how openseeface works and the computer vision models, go see https://github.com/emilianavt/OpenSeeFace I don't want to leave documentation here that makes it seem like this is all my work

Seriously, emilianavt did all the hard work

-----License-----
idk, I didn't plan to get this far
feel free to use my software and code for whatever, it'd be nice to get credit if you do
Just don't do anything to get me sued, or harass people, or make hateful content
Do use my software to be
