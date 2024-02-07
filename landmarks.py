import os
import numpy as np
import math
import cv2
import onnxruntime
import time

def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    return np.array(q) / (math.sqrt(t)* 0.5)

#I haven't touched much here
#not because I didn't try to improve things
#but because I caused problems when I did
class Landmarks():
    def __init__(self, width, height, threshold):

        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.width = width
        self.height = height
        self.confidence = 0

        self.face_3d = np.array([
            [ 0.4551769692672  ,  0.300895790030204, -0.764429433974752],
            [ 0.448998827123556,  0.166995837790733, -0.765143004071253],
            [ 0.437431554952677,  0.022655479179981, -0.739267175112735],
            [ 0.415033422928434, -0.088941454648772, -0.747947437846473],
            [ 0.389123587370091, -0.232380029794684, -0.704788385327458],
            [ 0.334630113904382, -0.361265387599081, -0.615587579236862],
            [ 0.263725112132858, -0.460009725616771, -0.491479221041573],
            [ 0.16241621322721 , -0.558037146073869, -0.339445180872282],
            [ 0.               , -0.621079019321682, -0.287294770748887],
            [-0.16241621322721 , -0.558037146073869, -0.339445180872282],
            [-0.263725112132858, -0.460009725616771, -0.491479221041573],
            [-0.334630113904382, -0.361265387599081, -0.615587579236862],
            [-0.389123587370091, -0.232380029794684, -0.704788385327458],
            [-0.415033422928434, -0.088941454648772, -0.747947437846473],
            [-0.437431554952677,  0.022655479179981, -0.739267175112735],
            [-0.448998827123556,  0.166995837790733, -0.765143004071253],
            [-0.4551769692672  ,  0.300895790030204, -0.764429433974752],
            [ 0.385529968662985,  0.402800553948697, -0.310031082540741],
            [ 0.322196658344302,  0.464439136821772, -0.250558059367669],
            [ 0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
            [ 0.186875436782135,  0.44706071961879 , -0.145299823706503],
            [ 0.120880983543622,  0.423566314072968, -0.110757158774771],
            [-0.120880983543622,  0.423566314072968, -0.110757158774771],
            [-0.186875436782135,  0.44706071961879 , -0.145299823706503],
            [-0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
            [-0.322196658344302,  0.464439136821772, -0.250558059367669],
            [-0.385529968662985,  0.402800553948697, -0.310031082540741],
            [ 0.               ,  0.293332603215811, -0.137582088779393],
            [ 0.               ,  0.194828701837823, -0.069158109325951],
            [ 0.               ,  0.103844017393155, -0.009151819844964],
            [ 0.               ,  0.               ,  0.               ],
            [ 0.080626352317973, -0.041276068128093, -0.134161035564826],
            [ 0.046439347377934, -0.057675223874769, -0.102990627164664],
            [ 0.               , -0.068753126205604, -0.090545348482397],
            [-0.046439347377934, -0.057675223874769, -0.102990627164664],
            [-0.080626352317973, -0.041276068128093, -0.134161035564826],
            [ 0.315905195966084,  0.298337502555443, -0.285107407636464],
            [ 0.275252345439353,  0.312721904921771, -0.244558251170671],
            [ 0.176394511553111,  0.311907184376107, -0.219205360345231],
            [ 0.131229723798772,  0.284447361805627, -0.234239149487417],
            [ 0.184124948330084,  0.260179585304867, -0.226590776513707],
            [ 0.279433549294448,  0.267363071770222, -0.248441437111633],
            [-0.131229723798772,  0.284447361805627, -0.234239149487417],
            [-0.176394511553111,  0.311907184376107, -0.219205360345231],
            [-0.275252345439353,  0.312721904921771, -0.244558251170671],
            [-0.315905195966084,  0.298337502555443, -0.285107407636464],
            [-0.279433549294448,  0.267363071770222, -0.248441437111633],
            [-0.184124948330084,  0.260179585304867, -0.226590776513707],
            [ 0.121155252430729, -0.208988660580347, -0.160606287940521],
            [ 0.041356305910044, -0.194484199722098, -0.096159882202821],
            [ 0.               , -0.205180167345702, -0.083299217789729],
            [-0.041356305910044, -0.194484199722098, -0.096159882202821],
            [-0.121155252430729, -0.208988660580347, -0.160606287940521],
            [-0.132325402795928, -0.290857984604968, -0.187067868218105],
            [-0.064137791831655, -0.325377847425684, -0.158924039726607],
            [ 0.               , -0.343742581679188, -0.113925986025684],
            [ 0.064137791831655, -0.325377847425684, -0.158924039726607],
            [ 0.132325402795928, -0.290857984604968, -0.187067868218105],
            [ 0.181481567104525, -0.243239316141725, -0.231284988892766],
            [ 0.083999507750469, -0.239717753728704, -0.155256465640701],
            [ 0.               , -0.256058040176369, -0.0950619498899  ],
            [-0.083999507750469, -0.239717753728704, -0.155256465640701],
            [-0.181481567104525, -0.243239316141725, -0.231284988892766],
            [-0.074036069749345, -0.250689938345682, -0.177346470406188],
            [ 0.               , -0.264945854681568, -0.112349967428413],
            [ 0.074036069749345, -0.250689938345682, -0.177346470406188],
            # Pupils and eyeball centers
            [ 0.257990002632141,  0.276080012321472, -0.219998998939991],
            [-0.257990002632141,  0.276080012321472, -0.219998998939991],
            [ 0.257990002632141,  0.276080012321472, -0.324570998549461],
            [-0.257990002632141,  0.276080012321472, -0.324570998549461]
        ], np.float32)

        self.contour_pts = [0,1,8,15,16,27,28,29,30,31,32,33,34,35]
        self.fail_count = 0


    def run(self, models, crop, crop_info):

        output = models.landmarks.run([], {"input": crop})[0][0]
        self.confidence, lms = self.landmarks(output, crop_info)
        return ( lms)

    def landmarks(self, tensor, crop_info):
        crop_x1, crop_y1, scale_x, scale_y = crop_info

        t_main = tensor[0:66].reshape((66,784))
        t_m = t_main.argmax(1)
        indices = np.expand_dims(t_m, 1)

        t_off_x = np.take_along_axis(tensor[66:132].reshape((66,784)), indices, 1).reshape((66,))
        p = np.clip(t_off_x, 0.0000001, 0.9999999)
        t_off_x =  13.9375 * np.log(p / (1 - p))
        t_x = crop_y1 + scale_y * (223. * np.floor(t_m / 28) / 27.+ t_off_x)

        t_off_y = np.take_along_axis(tensor[132:198].reshape((66,784)), indices, 1).reshape((66,))
        p = np.clip(t_off_y, 0.0000001, 0.9999999)
        t_off_y =  13.9375 * np.log(p / (1 - p))
        t_y = crop_x1 + scale_x * (223. * np.floor(np.mod(t_m, 28)) / 27. + t_off_y)

        t_conf = np.take_along_axis(t_main, indices, 1).reshape((66,))
        lms = np.stack([t_x, t_y, t_conf], 1)
        lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
        return (np.average(t_conf), lms)



    #I really need to break this guy up into smaller parts
    def estimate_depth(self, face_info):
        lms = np.concatenate((face_info.lms, np.array([[face_info.eye_state[0][1], face_info.eye_state[0][2], face_info.eye_state[0][3]], [face_info.eye_state[1][1], face_info.eye_state[1][2], face_info.eye_state[1][3]]], np.float32)), 0)
        image_pts = np.array(lms)[self.contour_pts, 0:2]

        success = False
        if face_info.rotation is not None:
            success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, self.camera, np.zeros((4,1)), useExtrinsicGuess=True, rvec=np.transpose(face_info.rotation), tvec=np.transpose(face_info.translation), flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            rvec = np.array([0, 0, 0], np.float32)
            tvec = np.array([0, 0, 0], np.float32)
            success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, self.camera, np.zeros((4,1)), useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            face_info.rotation = np.array([0.0, 0.0, 0.0], np.float32)
            face_info.translation = np.array([0.0, 0.0, 0.0], np.float32)
            return False, np.zeros(4), np.zeros(3), 99999., pts_3d, lms

        rotation = face_info.rotation
        translation = face_info.translation
        pts_3d = np.zeros((70,3), np.float32)

        face_info.rotation = np.transpose(face_info.rotation)
        face_info.translation = np.transpose(face_info.translation)

        rmat, _ = cv2.Rodrigues(rotation)
        inverse_rotation = np.linalg.inv(rmat)
        t_reference = face_info.face_3d.dot(rmat.transpose())
        t_reference = t_reference + face_info.translation
        t_reference = t_reference.dot(self.camera.transpose())
        t_depth = t_reference[:, 2]
        t_depth[t_depth == 0] = 0.000001
        t_depth_e = np.expand_dims(t_depth[:],1)
        t_reference = t_reference[:] / t_depth_e
        pts_3d[0:66] = np.stack([lms[0:66,0], lms[0:66,1], np.ones((66,))], 1) * t_depth_e[0:66]
        pts_3d[0:66] = (pts_3d[0:66].dot(self.inverse_camera.transpose()) - face_info.translation).dot(inverse_rotation.transpose())

        for i, pt in enumerate(face_info.face_3d[66:70]):
            if i == 2:
                # Right eyeball
                # Eyeballs have an average diameter of 12.5mm and and the distance between eye corners is 30-35mm, so a conversion factor of 0.385 can be applied
                eye_center = (pts_3d[36] + pts_3d[39]) / 2.0
                d_corner = np.linalg.norm(pts_3d[36] - pts_3d[39])
                depth = 0.385 * d_corner
                pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
                pts_3d[68] = pt_3d
                continue
            if i == 3:
                # Left eyeball
                eye_center = (pts_3d[42] + pts_3d[45]) / 2.0
                d_corner = np.linalg.norm(pts_3d[42] - pts_3d[45])
                depth = 0.385 * d_corner
                pt_3d = np.array([eye_center[0], eye_center[1], eye_center[2] - depth])
                pts_3d[69] = pt_3d
                continue
            if i == 0:
                d1 = np.linalg.norm(lms[66,0:2] - lms[36,0:2])
                d2 = np.linalg.norm(lms[66,0:2] - lms[39,0:2])
                d = d1 + d2
                pt = (pts_3d[36] * d1 + pts_3d[39] * d2) / d
            if i == 1:
                d1 = np.linalg.norm(lms[67,0:2] - lms[42,0:2])
                d2 = np.linalg.norm(lms[67,0:2] - lms[45,0:2])
                d = d1 + d2
                pt = (pts_3d[42] * d1 + pts_3d[45] * d2) / d
            if i < 2:
                reference = rmat.dot(pt)
                reference = reference + face_info.translation
                reference = self.camera.dot(reference)
                depth = reference[2]
                pt_3d = np.array([lms[66+i][0] * depth, lms[66+i][1] * depth, depth], np.float32)
                pt_3d = self.inverse_camera.dot(pt_3d)
                pt_3d = pt_3d - face_info.translation
                pt_3d = inverse_rotation.dot(pt_3d)
                pts_3d[66+i,:] = pt_3d[:]

        pts_3d[np.isnan(pts_3d).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
        pnp_error = np.power(lms[0:17,0:2] - t_reference[0:17,0:2], 2).sum()
        pnp_error += np.power(lms[30,0:2] - t_reference[30,0:2], 2).sum()
        if np.isnan(pnp_error):
            pnp_error = 9999999.
        pnp_error = math.sqrt(pnp_error / (2.0 * image_pts.shape[0]))
        if pnp_error > 300:
            self.fail_count += 1
            if self.fail_count > 5:
                # Something went wrong with adjusting the 3D model
                print(f"Detected anomaly when 3D fitting face {face_info.id}. Resetting.")
                face_info.face_3d = self.face_3d
                face_info.rotation = None
                face_info.translation = np.array([0.0, 0.0, 0.0], np.float32)
                face_info.update_counts = np.zeros((66,2))
                face_info.update_contour()
        else:
            self.fail_count = 0

        euler = cv2.RQDecomp3x3(rmat)[0]
        return True, matrix_to_quaternion(rmat), euler, pnp_error, pts_3d, lms
