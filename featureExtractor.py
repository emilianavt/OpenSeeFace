import feature
import math

def rotate(origin, point, a):
    x, y = point - origin
    cosa = math.cos(-a)
    sina = math.sin(-a)
    qx = origin[0] + cosa * x - sina * y
    qy = origin[1] + sina * x + cosa * y
    return qx, qy


#This class determines how various parameters operate, I'l do my best to document it here so you can customize them
#It's not a replacement for vbridger, but it'll get you some level of control beyond what Vtube Studio offers




class FeatureExtractor():
    def __init__(self):
        #These define how the normalization in feature.py works
        #For each feature you can pass in alpha, decay, curve, and scaleType

        #alpha is how much much of the previous value is mixed in with the current value
        #it's on a scale of 0 to 1, the default of 0.1 is 10%
        #Making this higher makes things slower, but smoother
        #Making it lower makes things more reactive but prone to jitter
        #I like to have it low which is why the default is 0.1

        #decay is how fast the upper and lower bounds of the parameters move towards their center value
        #This is part of how I handle calibration
        #It means the range for that paramete is constantly moving to match you, and if the calibration is off it doesn't stay that way
        #If you don't like the sound of this, set the decay to 0
        #Generally this will be a very very small number, the default is 0.0005
        #A value nearing 1 would always have you setting new limits
        #A value of 1 might cause a crash

        #curve determines how that parameter reacts
        #It's just an exponent applied to the normalized value, which ranges from 0 to 1 (or -1 to 0 to 1)
        #scaleType 1 (the default) handles this in a way that's kinda both intuitive and unintuitive
        #It applies the curve separately to the positive and negative sides of the parameter
        #this means cuves lower than 1 work without making imaginary numbers
        #You may find it helpful to graph y=x^[curve] on a graphing calulator such as Desmos https://www.desmos.com/calculator
        #then zoom in on the 0-1 range to get an idea of how these response cuves work

        #scaleType is mostly determined by how Vtube Studio handles a parameter
        #scaleType = 1 has a range of -1 to 1, with 0 being the center between the maximum and minimum values
        #scaleType = 2 has a range of 0 to 1
        #For values like Eye Openness and Mouth Openness I reccomend scaleType = 2
        #for Eyebrows and Mouth Corners I reccomend 1


        self.eye_l = feature.Feature(scaleType = 2, curve = 0.5)
        self.eye_r = feature.Feature(scaleType = 2, curve = 0.5)
        self.eyebrow_updown_l = feature.Feature( curve = 1.5)
        self.eyebrow_updown_r = feature.Feature( curve = 1.5)
        self.mouth_corner_updown_l = feature.Feature(curve = 2, decay=0.00005  )
        self.mouth_corner_updown_r = feature.Feature(curve = 2, decay=0.00005 )
        self.mouth_open = feature.Feature(scaleType = 2, curve = 1.5)




    #This is the arbitrary messy math portion of how parameters work
    #most of the stuff here was set by trial and error , so it's kinda weird
    #the numbers in pts coorespond to the iBug 68 face landmarks
    #The f values are what get sent to the feature normalization class
    #anything after the .update(f) method is called changes what is sent to Vtube Studio directly
    #the align_points() method rotates the given points such that the first two points line up
    #you can really go wild with this and even make things work in strange ways
    #like making your eyes open your mouth or something


    def update(self, pts):
        features = {}
        norm_distance_y = pts[27, 1]-pts[8, 1]

        f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        f = (f_pts[0][1]/2 + f_pts[1][1]/2) - (f_pts[2][1]/2 + f_pts[3][1]/2)
        features["eye_l"] =  self.eye_l.update(f)
        f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        f = (f_pts[0][1]/2 + f_pts[1][1]/2) - (f_pts[2][1]/2 + f_pts[3][1]/2)
        features["eye_r"] = self.eye_r.update(f)

        f = ((pts[22][1]/2 + pts[26][1]/2) - (pts[42][1]/2 + pts[45][1]/2))/norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f)
        f = ((pts[17][1]/2 + pts[21][1]/2) - (pts[36][1]/2 + pts[39][1]/2))/norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f)

        f = ( pts[50][1] - pts[55][1]) * (norm_distance_y)
        features["mouth_open"] = self.mouth_open.update(f)*0.66

        #mouth corners are calculated together so they are assigned the same value
        f = (((pts[51][1] + pts[55][1])/4) - ((pts[58][1]+pts[62][1])/2))/ norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f) * (1- 0.3 *features["mouth_open"]) * 0.66 - 0.1
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f) * (1- 0.3 * features["mouth_open"]) *0.66 - 0.1

        #i removed some features VTS didn't seem to use
        #the VTS library was handling them not being set anyway, so I just let that deal with the fact they're not here
        return features


    def align_points(self, a, b, pts): #runs 6 times per frame

        alpha = (math.atan2(*(b - a)[::-1]) % (math.pi *2))

        if alpha >= (math.pi/2):
            alpha = - (alpha - math.pi)
        if alpha <= -(math.pi/2):
            alpha = - (alpha + math.pi)
        aligned_pts = []
        for pt in pts:
            aligned_pts.append(rotate(a, pt, alpha))
        return (aligned_pts)
