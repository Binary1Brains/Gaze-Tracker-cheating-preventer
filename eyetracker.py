import cv2
import mediapipe as mp
import numpy as np

horizontal_threshold = 2.5
horizontal_movement_counter = 0
horizontal_movement_threshold = 100
p=0

draw_gaze = True
draw_full_axis = True
draw_headpose = False

x_score_multiplier = 10
y_score_multiplier = 10

threshold = .3
reading_counter = 0
reading_threshold = 10

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

face_3d = np.array([[0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]], dtype=np.float64)

leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0


while cap.isOpened():
    success, img = cap.read()

    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    img.flags.writeable = False
    
    results = face_mesh.process(img)
    img.flags.writeable = True

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

    if not results.multi_face_landmarks:
      continue 

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append((x, y))
        
        face_2d_head = np.array([face_2d[1], face_2d[199], face_2d[33], face_2d[263], face_2d[61], face_2d[291]], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier


        l_corner = face_2d_head[2].astype(np.int32)
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        r_corner = face_2d_head[3].astype(np.int32)

        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
        r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)
        screen_width = img_w
        screen_height = img_h

        left_region = (0, 0, screen_width // 3, screen_height) 
        center_region = (screen_width // 3, 0, screen_width // 3, screen_height) 
        right_region = (2 * screen_width // 3, 0, screen_width // 3, screen_height) 
        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)
    if abs(l_gaze_rvec[0][0]) > horizontal_threshold and abs(r_gaze_rvec[0][0]) > horizontal_threshold:
              horizontal_movement_counter += 1
    else:
              horizontal_movement_counter = 0
    if horizontal_movement_counter >= horizontal_movement_threshold:
              p+=1
    
    if (p>100):
        p=0;
        print ("The candidate might be cheating. Please pay attention ")  
    cv2.imshow('Head Position Estimation', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()