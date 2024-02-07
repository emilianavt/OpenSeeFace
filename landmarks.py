import numpy as np
import math
import cv2

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

def landmarks(tensor, crop_info):
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

def run( models, crop, crop_info):

    output = models.landmarks.run([], {"input": crop})[0][0]
    confidence, lms = landmarks(output, crop_info)
    return (confidence, lms)


def estimate_depth( face_info, width, height):
    camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
    inverse_camera = np.linalg.inv(camera)
    contour_pts = [0,1,8,15,16,27,28,29,30,31,32,33,34,35]
    lms = np.concatenate((face_info.lms, np.array([[face_info.eye_state[0][1], face_info.eye_state[0][2], face_info.eye_state[0][3]], [face_info.eye_state[1][1], face_info.eye_state[1][2], face_info.eye_state[1][3]]], np.float32)), 0)
    image_pts = np.array(lms)[contour_pts, 0:2]

    success = False
    if face_info.rotation is not None:
        success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, camera, np.zeros((4,1)), useExtrinsicGuess=True, rvec=np.transpose(face_info.rotation), tvec=np.transpose(face_info.translation), flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        rvec = np.array([0, 0, 0], np.float32)
        tvec = np.array([0, 0, 0], np.float32)
        success, face_info.rotation, face_info.translation = cv2.solvePnP(face_info.contour, image_pts, camera, np.zeros((4,1)), useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv2.SOLVEPNP_ITERATIVE)

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
    t_reference = t_reference.dot(camera.transpose())
    t_depth = t_reference[:, 2]
    t_depth[t_depth == 0] = 0.000001
    t_depth_e = np.expand_dims(t_depth[:],1)
    t_reference = t_reference[:] / t_depth_e
    pts_3d[0:66] = np.stack([lms[0:66,0], lms[0:66,1], np.ones((66,))], 1) * t_depth_e[0:66]
    pts_3d[0:66] = (pts_3d[0:66].dot(inverse_camera.transpose()) - face_info.translation).dot(inverse_rotation.transpose())

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
            reference = camera.dot(reference)
            depth = reference[2]
            pt_3d = np.array([lms[66+i][0] * depth, lms[66+i][1] * depth, depth], np.float32)
            pt_3d = inverse_camera.dot(pt_3d)
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
        face_info.fail_count += 1
        if face_info.fail_count > 5:
            # Something went wrong with adjusting the 3D model
            print(f"Detected anomaly when 3D fitting face {0}. Resetting.")
            face_info.rotation = None
            face_info.translation = np.array([0.0, 0.0, 0.0], np.float32)
            face_info.update_counts = np.zeros((66,2))
            face_info.update_contour()
    else:
        face_info.fail_count = 0

    euler = cv2.RQDecomp3x3(rmat)[0]
    return True, matrix_to_quaternion(rmat), euler, pnp_error, pts_3d, lms
