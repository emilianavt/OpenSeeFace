import numpy as np
import cv2





def detect_faces(frame, model, detection_threshold ):

        image = resizeImage(frame)

        outputs, _ = model.faceDetection.run([], {'input': image})
        outputs = outputs[0]
        faceLocation = np.argmax(outputs[0].flatten())
        x = faceLocation % 56
        y = faceLocation // 56

        if outputs[0, y, x] < detection_threshold:
            return None

        r = outputs[1, y, x] * 112.
        results= (((x * 4) - r, (y * 4) - r, r*2,r*2))
        results = np.array(results).astype(np.float32)
        results[[0,2]] *= frame.shape[1] / 224.
        results[[1,3]] *= frame.shape[0] / 224.

        return results

def resizeImage(frame):
    mean = np.float32(np.array([-2.1179, -2.0357, -1.8044]))
    std = np.float32(np.array([0.0171, 0.0175, 0.0174]))
    targetimageSize = [224, 224]
    image = cv2.resize(frame, targetimageSize, interpolation=cv2.INTER_LINEAR) * std + mean
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0,3,1,2))
    return image



