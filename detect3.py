from ultralytics import YOLO
import cv2
import time

# ==========================================
# Helper Functions
# ==========================================

def is_valid_chute(truck_box, chute_box):
    truck_y1 = truck_box[1]
    chute_y1 = chute_box[1]

    if chute_y1 < truck_y1:
        return False

    expanded_tx1 = truck_box[0] - 100  # ลดจาก 200 → 100 เหมาะกับ resolution ต่ำ

    ix1 = max(expanded_tx1, chute_box[0])
    iy1 = max(truck_box[1], chute_box[1])
    ix2 = min(truck_box[2], chute_box[2])
    iy2 = min(truck_box[3], chute_box[3])

    if ix2 < ix1 or iy2 < iy1:
        return False

    return True

def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def is_truck_still(current_box, last_box, move_threshold):
    if last_box is None:
        return False
    cx1, cy1 = get_center(current_box)
    cx2, cy2 = get_center(last_box)
    moved = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
    return moved < move_threshold

# ==========================================
# Configuration
# ==========================================

MODEL_PATH    = r"C:\Users\HP\OneDrive\Desktop\Yolotest3\runs\detect\concretemix_v8\weights\best.pt"
CAMERA_SOURCE = "3D54CCC96C4C1915_2026-06-15T06-30-01-468Z.webm"

# --- Drop Point ---
# กล้องจริง 640x360 — คลิกซ้ายบนจอเพื่อ calibrate แล้วดูค่าใน terminal
DROP_POINT_X = 236    # เลื่อนจาก 160 → 236 (ใกล้จุดสีเหลือง, fine-tune ด้วยการ click)
DROP_POINT_Y = 215    # ปรับตาม Y จริง (fine-tune ด้วยการ click)
DROP_RADIUS  = 60     # เพิ่มจาก 40 → 60 (กล้อง 640px)

# --- Classification ---
# กล้อง 640x360: small truck h≈250-255, big truck h≈285+
# HEIGHT_THRESHOLD ต้องอยู่ระหว่าง 255 กับ 285
HEIGHT_THRESHOLD = 270     # small truck h≈250-255 < 270 = SMALL TRUCK ✓
                           # big truck   h≈285+   > 270 = BIG TRUCK  ✓
LOCK_MIN_AREA    = 30_000  # ยังใช้ได้ (area ~104,000 >> 30,000)
VOTES_NEEDED     = 5

# --- Timing thresholds ---
MOVE_THRESHOLD        = 60   # เพิ่มจาก 40 → 60 (กล้อง 640px)
MAX_MISSING_FRAMES    = 90
REQUIRED_STILL_FRAMES = 30
REQUIRED_OUT_FRAMES   = 30
MAX_DEPARTING_FRAMES  = 60

# ==========================================
# Mouse Calibration Callback
# ==========================================

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[CALIBRATE] คลิกที่ x={x}, y={y}  →  ตั้ง DROP_POINT_X={x}, DROP_POINT_Y={y}")

# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(CAMERA_SOURCE)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAMERA] Resolution: {cam_w} x {cam_h}")
    print(f"[CONFIG] DROP=({DROP_POINT_X},{DROP_POINT_Y}) radius={DROP_RADIUS} | "
          f"HEIGHT_THRESH={HEIGHT_THRESHOLD} | MIN_AREA={LOCK_MIN_AREA}")

    # --- State ---
    current_state     = "EMPTY"
    driver_action     = ""
    locked_truck_type = None

    # --- Counters ---
    missing_frames     = 0
    still_frames       = 0
    out_of_zone_frames = 0
    departing_frames   = 0
    frame_count        = 0

    # --- Vote buffer ---
    type_votes     = {"BIG TRUCK": 0, "SMALL TRUCK": 0}
    last_truck_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=0.3, verbose=False)

        truck_box      = None
        chute_box      = None
        max_truck_conf = 0
        max_chute_conf = 0

        for result in results:
            for box in result.boxes:
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_name == "cement_mixer_truck" and conf > max_truck_conf:
                    truck_box      = (x1, y1, x2, y2, conf)
                    max_truck_conf = conf
                elif cls_name == "swing_chute" and conf > max_chute_conf:
                    chute_box      = (x1, y1, x2, y2, conf)
                    max_chute_conf = conf

        truck_found = truck_box is not None

        if truck_found and chute_box is not None:
            if not is_valid_chute(truck_box, chute_box):
                chute_box = None

        in_zone            = False
        is_still           = False
        display_truck_type = "UNKNOWN"
        truck_area         = 0
        truck_w = truck_h  = 0
        target_x = target_y = 0
        distance_to_drop   = 9999

        # ==========================================
        # 1. Coordinate Analysis + Classification
        # ==========================================
        if truck_found:
            missing_frames = 0
            tx1, ty1, tx2, ty2, conf = truck_box

            truck_w    = tx2 - tx1
            truck_h    = ty2 - ty1
            truck_area = truck_w * truck_h

            if locked_truck_type is None and truck_area > LOCK_MIN_AREA:
                if chute_box is not None:
                    vote = "SMALL TRUCK"
                else:
                    vote = "BIG TRUCK" if truck_h > HEIGHT_THRESHOLD else "SMALL TRUCK"

                type_votes[vote] += 1
                other = "SMALL TRUCK" if vote == "BIG TRUCK" else "BIG TRUCK"
                type_votes[other] = max(0, type_votes[other] - 1)

                if type_votes[vote] >= VOTES_NEEDED:
                    locked_truck_type = vote
                    type_votes = {"BIG TRUCK": 0, "SMALL TRUCK": 0}
                    print(f"[LOCKED] Type = {locked_truck_type} | area={truck_area} h={truck_h}")

            elif locked_truck_type == "BIG TRUCK" and chute_box is not None:
                type_votes["SMALL TRUCK"] += 1
                if type_votes["SMALL TRUCK"] >= 3:
                    locked_truck_type = "SMALL TRUCK"
                    type_votes = {"BIG TRUCK": 0, "SMALL TRUCK": 0}

            display_truck_type = locked_truck_type if locked_truck_type else "DETECTING..."

            if display_truck_type == "SMALL TRUCK":
                if chute_box is not None:
                    cx1, cy1, cx2, cy2, _ = chute_box
                    target_x = int((cx1 + cx2) / 2)
                else:
                    target_x = tx1
            elif display_truck_type == "BIG TRUCK":
                target_x = tx1 + int(truck_w * 0.15)  # เพิ่มจาก 10% → 20% (ขยับจุดเหลืองเข้าไปอีก)
            else:
                target_x = tx1

            target_y         = DROP_POINT_Y
            distance_to_drop = abs(target_x - DROP_POINT_X)
            in_zone          = distance_to_drop <= DROP_RADIUS

            current_truck_box = truck_box[:4]
            is_still          = is_truck_still(current_truck_box, last_truck_box, MOVE_THRESHOLD)
            last_truck_box    = current_truck_box

        else:
            missing_frames += 1

        # ==========================================
        # 2. State Machine
        # ==========================================
        if missing_frames > MAX_MISSING_FRAMES:
            current_state      = "EMPTY"
            locked_truck_type  = None
            driver_action      = ""
            still_frames       = 0
            out_of_zone_frames = 0
            departing_frames   = 0
            last_truck_box     = None
            type_votes         = {"BIG TRUCK": 0, "SMALL TRUCK": 0}

        else:
            if current_state == "EMPTY" and truck_found:
                current_state      = "ARRIVED"
                still_frames       = 0
                out_of_zone_frames = 0
                departing_frames   = 0

            elif current_state == "ARRIVED":
                if in_zone:
                    still_frames  = still_frames + 1 if is_still else max(0, still_frames - 2)
                    driver_action = "PLEASE STOP TRUCK"
                    if still_frames >= REQUIRED_STILL_FRAMES:
                        current_state      = "LOADING"
                        out_of_zone_frames = 0
                else:
                    still_frames = max(0, still_frames - 2)
                    if target_x > (DROP_POINT_X + DROP_RADIUS):
                        driver_action = "<< PLEASE MOVE BACK"
                    elif target_x < (DROP_POINT_X - DROP_RADIUS):
                        driver_action = "MOVE FORWARD >>"
                    else:
                        driver_action = "ADJUST POSITION"

            elif current_state == "LOADING":
                driver_action = "STOP & LOAD"
                if truck_found:
                    if in_zone:
                        out_of_zone_frames = max(0, out_of_zone_frames - 1)
                    else:
                        out_of_zone_frames += 1
                    if out_of_zone_frames >= REQUIRED_OUT_FRAMES:
                        current_state      = "DEPARTING"
                        out_of_zone_frames = 0
                        departing_frames   = 0
                elif missing_frames > 30:
                    current_state    = "DEPARTING"
                    departing_frames = 0

            elif current_state == "DEPARTING":
                driver_action     = "CLEAR TO LEAVE"
                departing_frames += 1

                if in_zone and is_still:
                    current_state      = "LOADING"
                    out_of_zone_frames = 0
                    still_frames       = REQUIRED_STILL_FRAMES
                    departing_frames   = 0
                elif departing_frames > MAX_DEPARTING_FRAMES:
                    current_state     = "EMPTY"
                    locked_truck_type = None
                    driver_action     = ""
                    departing_frames  = 0

        # ==========================================
        # 3. Terminal Log
        # ==========================================
        if frame_count % 5 == 0 and truck_found:
            print(
                f"[Frame {frame_count:04d}] "
                f"Type: {display_truck_type:<11} | "
                f"State: {current_state:<9} | "
                f"Dist: {distance_to_drop:.1f}px | "
                f"h:{truck_h} area:{truck_area} | "
                f"Action: {driver_action}"
            )

        # ==========================================
        # 4. Draw UI
        # ==========================================
        height, width, _ = frame.shape
        font_scale = 0.5 if width <= 400 else 0.8

        # Drop target circle
        cv2.circle(frame, (DROP_POINT_X, DROP_POINT_Y), DROP_RADIUS, (255, 0, 255), 2)
        cv2.drawMarker(frame, (DROP_POINT_X, DROP_POINT_Y), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(frame, "DROP TARGET",
                    (max(0, DROP_POINT_X - 45), max(12, DROP_POINT_Y - DROP_RADIUS - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        if truck_found:
            if display_truck_type == "SMALL TRUCK":
                box_color = (0, 255, 0)
            elif display_truck_type == "BIG TRUCK":
                box_color = (255, 0, 0)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), box_color, 2)
            cv2.putText(frame, display_truck_type,
                        (tx1, max(12, ty1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            if chute_box is not None:
                cx1, cy1, cx2, cy2, _ = chute_box
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
                cv2.putText(frame, "CHUTE", (cx1, max(12, cy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            cv2.circle(frame, (target_x, target_y), 5, (0, 255, 255), -1)
            line_color = (0, 255, 0) if in_zone else (0, 0, 255)
            cv2.line(frame, (target_x, target_y), (DROP_POINT_X, DROP_POINT_Y), line_color, 2, cv2.LINE_AA)
            mid_x = int((target_x + DROP_POINT_X) / 2)
            mid_y = int((target_y + DROP_POINT_Y) / 2)
            cv2.putText(frame, f"{int(distance_to_drop)}px", (mid_x, max(12, mid_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

        # Status bar
        if current_state == "LOADING":
            ui_text, ui_color, action_color = f"READY TO LOAD [{display_truck_type}]", (255, 0, 255), (0, 255, 0)
        elif current_state == "ARRIVED":
            ui_text, ui_color, action_color = f"WAITING [{display_truck_type}]", (0, 255, 255), (0, 0, 255)
        elif current_state == "DEPARTING":
            ui_text, ui_color, action_color = f"COMPLETED [{display_truck_type}]", (255, 255, 0), (255, 255, 0)
        else:
            ui_text, ui_color, action_color = "NO TRUCK AT ZONE", (0, 0, 255), (0, 0, 255)

        cv2.putText(frame, ui_text, (5, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, ui_color, 2)

        truck_h_val = truck_h if truck_found else 0
        cv2.putText(frame,
            f"Miss:{missing_frames}/{MAX_MISSING_FRAMES} | Out:{out_of_zone_frames}/{REQUIRED_OUT_FRAMES} | "
            f"Still:{still_frames}/{REQUIRED_STILL_FRAMES} | Votes:{type_votes} | h:{truck_h_val}",
            (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if driver_action:
            blink    = int(time.time() * 2) % 2 == 0
            action_y = min(70, height // 5)
            if driver_action == "PLEASE STOP TRUCK":
                if blink:
                    cv2.putText(frame, driver_action, (5, action_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, action_color, 3)
            else:
                cv2.putText(frame, driver_action, (5, action_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, action_color, 3)

        cv2.imshow("CPAC Smart Loading Zone (Height System)", frame)
        cv2.setMouseCallback("CPAC Smart Loading Zone (Height System)", on_mouse)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
