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


For documentation on how openseeface works and the computer vision models, go see https://github.com/emilianavt/OpenSeeFace
I don't want to leave documentation here that makes it seem like this is all my work


Seriously, emilianavt did all the hard work


actual documentation is in the README.txt because working in plain text is easier than making github's markdown cooperate
