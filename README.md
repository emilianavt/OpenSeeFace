This is kind of a personal project, but it spiraled out of control and I figured I'd share it in case anyone else can make use of it.

I make no promises to the quality of this code or the reliablity of it

I am not responsible if this somehow causes you issues

I have drastically restructured this, but I'm listing it as a fork because this is still based on emilianavt's work.

I am specifically targeting Vtube Studio on Linux, some stuff was changed to make it work with VTS, idk how well it works with VeeSeeFace at this point

This is also very much tuned to how I use it and what works best for me, so I make no promises there

If you do use it, be aware that I removed directshow support(can't make use of it) and MJPEG is kinda janky, so your best bet is 480p which defaults to raw rgb

This will do a rock steady 30fps on my system as long as my cpu isn't completely overloaded


I'll update this readme to explain more later, but first I need to actually upload all my changes. I didn't expect this to turn into something I wanted to upload. 

Packages you'll need

Numpy

OpenCV-python

onnxruntime
<br>
<br>

# Changes (in no particular order):<br>
I tried to make this nice but indents don't exist here I guess<br>
-Major restructure<br>
-I've broken out major chunks of functionality into separate files to make them easier to deal with and read<br>
---I've tried to break functions down into multiple smaller parts when possible<br>
---I've taken a more object oriented approach in some places<br>
---I've removed features that didn't fit my needs<br>
---Removed functionlaity for tracking multiple faces<br>
---Removed options I didn't forsee myself using (this was intended to be just for me)<br>
-Added more threading and multi-processing<br>
---The webcam (and some new image processing) are now on a separate process<br>
---Previews are now a separate process because I didn't want to deal with the related performance weirdness<br>
---Console messages are now handled via a helper thread, idk if it does any good, but it can't hurt<br>
---VTS communication is also handled via a helper thread<br>
-Added image pre-processing<br>
---I'm now applying a gamma curve to the webcam output to make faces more visible<br>
---The gamma curve is calculated after face tracking is done, it uses a copy of the webcam frame and face location data so get the average   brightness of the face<br>
---This has drasitcally improved low light performance<br>
---It also lets me run the webcam with no gain<br>
---Removed the separate library that handled the webcam, then broke that off into a separate library again<br>
---The stuff around the webcam is still simpler now, most of the existing cases didn't apply to my use case<br>
-Revamped the way features are calculated and normalized<br>
---This was kind of my original intent, my eyes kept registering as closed when they were not<br>
---Removed features that didn't seem to be used by Vtube Studio<br>
---Added functionality to apply response curves to features<br>
---Removed the calibration period in favor of a system where the limits of features slowly decay towards a center point<br>
---Removed the average from the way featured are normalized<br>
-Added a feature to prevent errant eye movements<br>
---Each eye has an average confidence and standard deviation calculated every frame<br>
---eye movements more than two standard deviations below average are severely restricted<br>
---The restriction is based on how far the confidence is from the cutoff point<br>
-Added early exits to situations where subsequent steps will fail<br>
-The main process skips frames when the webcam falls behind, once webcam latency is long enough to be late it will remain behind otherwise<br>
-Added warnings to various states such as late webcam frames, longer than ideal frames, and early exits<br>
-Changed which stats are shown on exit, added some stats that get tracked like webcam latency<br>
-Added conditions to skip steps when they are unnecessary, such as not rotating the eye images for small angles<br>
<br>
I'll list more as I remember them<br>
<br>

For documentation on how openseeface works and the computer vision models, go see https://github.com/emilianavt/OpenSeeFace
I don't want to leave documentation here that makes it seem like this is all my work


Seriously, emilianavt did all the hard work
