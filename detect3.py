from ultralytics import YOLO
import cv2
import time

# ==========================================
# Helper Functions
# ==========================================

def is_valid_chute(truck_box, chute_box):
    """
    [Ghost Filter & Swing Chute Revival]
    Filters out plant discharge pipes, and allows swing chutes to extend beyond the rear of the truck.
    """
    truck_y1 = truck_box[1]
    chute_y1 = chute_box[1]

    # Rule 1: Plant discharge pipes hang from the ceiling (y1 near the top edge).
    # If the swing chute is higher than the truck roof, discard it as a ghost immediately.
    if chute_y1 < truck_y1:
        return False

    # Rule 2: The swing chute can extend outside the truck (left side)!
    # Expand the detection area behind the truck by 200 pixels.
    expanded_tx1 = truck_box[0] - 200 
    
    ix1 = max(expanded_tx1, chute_box[0])
    iy1 = max(truck_box[1], chute_box[1])
    ix2 = min(truck_box[2], chute_box[2])
    iy2 = min(truck_box[3], chute_box[3])

    # If there's still no overlap even after expanding 200px, then discard.
    if ix2 < ix1 or iy2 < iy1:
        return False
        
    return True

def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def is_truck_still(current_box, last_box, move_threshold=80):
    if last_box is None:
        return False
    cx1, cy1 = get_center(current_box)
    cx2, cy2 = get_center(last_box)
    moved = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
    return moved < move_threshold

# ==========================================
# Configuration (Bullseye System)
# ==========================================

MODEL_PATH     = r"C:\Users\HP\OneDrive\Desktop\Yolotest3\runs\detect\concretemix_v6_new-2\weights\best.pt"
CAMERA_SOURCE  = "_Camera 01_20260611132433_59073164.mp4"

# --- Drop Point (Plant discharge pipe target) ---
DROP_POINT_X = 580    # Center X coordinate of the plant pipe (adjustable per camera)
DROP_POINT_Y = 450    # Center Y coordinate of the plant pipe
DROP_RADIUS  = 80     # Acceptance radius (pixels) — within this distance = on target

# --- Classification thresholds (Upgraded system: uses truck height to decide) ---
HEIGHT_THRESHOLD = 450       # Minimum height (pixels) — if truck exceeds this = definitely a big truck!
LOCK_MIN_AREA    = 200_000   # Wait until the truck bounding box is large enough before starting votes
VOTES_NEEDED     = 5         # Number of consecutive votes needed to lock the truck type

# --- Timing thresholds (frames) ---
MOVE_THRESHOLD       = 80
MAX_MISSING_FRAMES   = 90
REQUIRED_STILL_FRAMES = 30
REQUIRED_OUT_FRAMES  = 30    
MAX_DEPARTING_FRAMES = 60   

# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(CAMERA_SOURCE)

    # --- State ---
    current_state    = "EMPTY"
    driver_action    = ""
    locked_truck_type = None

    # --- Counters ---
    missing_frames     = 0
    still_frames       = 0
    out_of_zone_frames = 0
    departing_frames   = 0
    frame_count        = 0

    # --- Vote buffer ---
    type_votes   = {"BIG TRUCK": 0, "SMALL TRUCK": 0}
    last_truck_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=0.3, verbose=False)
        
        truck_box = None
        chute_box = None
        max_truck_conf = 0
        max_chute_conf = 0

        # Loop and separate by class
        for result in results:
            for box in result.boxes:
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_name == "cement_mixer_truck" and conf > max_truck_conf:
                    truck_box = (x1, y1, x2, y2, conf)
                    max_truck_conf = conf
                elif cls_name == "swing_chute" and conf > max_chute_conf:
                    chute_box = (x1, y1, x2, y2, conf)
                    max_chute_conf = conf

        truck_found = truck_box is not None
        
        # --- Filter ghost plant pipes ---
        if truck_found and chute_box is not None:
            if not is_valid_chute(truck_box, chute_box):
                chute_box = None 

        in_zone     = False
        is_still    = False
        display_truck_type = "UNKNOWN"
        truck_area  = 0
        truck_w = truck_h = 0
        
        target_x = 0
        target_y = 0
        distance_to_drop = 9999 

        # ==========================================
        # 1. Coordinate Analysis + Classification
        # ==========================================
        if truck_found:
            missing_frames = 0
            tx1, ty1, tx2, ty2, conf = truck_box

            truck_w      = tx2 - tx1
            truck_h      = ty2 - ty1
            truck_area   = truck_w * truck_h

            # --- Hybrid Vote System (Upgraded: uses truck height to decide) ---
            if locked_truck_type is None and truck_area > LOCK_MIN_AREA:
                if chute_box is not None:
                    vote = "SMALL TRUCK"
                else:
                    # If truck height exceeds HEIGHT_THRESHOLD, classify as big truck immediately
                    if truck_h > HEIGHT_THRESHOLD:
                        vote = "BIG TRUCK"
                    else:
                        vote = "SMALL TRUCK"
                
                type_votes[vote] += 1
                other = "SMALL TRUCK" if vote == "BIG TRUCK" else "BIG TRUCK"
                type_votes[other] = max(0, type_votes[other] - 1)
                
                if type_votes[vote] >= VOTES_NEEDED:
                    locked_truck_type = vote
                    type_votes = {"BIG TRUCK": 0, "SMALL TRUCK": 0}

            # --- Override: switch back to small truck if chute appears ---
            elif locked_truck_type == "BIG TRUCK" and chute_box is not None:
                type_votes["SMALL TRUCK"] += 1
                if type_votes["SMALL TRUCK"] >= 3: 
                    locked_truck_type = "SMALL TRUCK"
                    type_votes = {"BIG TRUCK": 0, "SMALL TRUCK": 0}

            display_truck_type = locked_truck_type if locked_truck_type else "DETECTING..."

            # --- Target Point calculation (Upgraded with offset compensation) ---
            if display_truck_type == "SMALL TRUCK":
                if chute_box is not None:
                    # 1. Small truck with visible chute -> aim at center of swing chute (most reliable)
                    cx1, cy1, cx2, cy2, _ = chute_box
                    target_x = int((cx1 + cx2) / 2)
                else:
                    # 2. Small truck, chute not visible -> attach yellow target to the left edge of bounding box (truck rear), no more pushing out!
                    target_x = tx1 
                    
            elif display_truck_type == "BIG TRUCK":
                # 3. Big truck, no chute -> pull target 10% inward from the left edge (aligns with concrete funnel position)
                target_x = tx1 + int(truck_w * 0.10)
                
            else:
                target_x = tx1 

            # Lock target Y to the same level as the pipe at all times
            # So the distance line stays horizontal, measuring only forward/backward movement
            target_y = DROP_POINT_Y 

            # --- Calculate distance to plant pipe ---
            distance_to_drop = ((target_x - DROP_POINT_X)**2 + (target_y - DROP_POINT_Y)**2)**0.5
            
            # If within radius = on target (In Zone)
            in_zone = distance_to_drop <= DROP_RADIUS

            current_truck_box = truck_box[:4]
            is_still = is_truck_still(current_truck_box, last_truck_box, MOVE_THRESHOLD)
            last_truck_box = current_truck_box

        else:
            missing_frames += 1

        # ==========================================
        # 2. State Machine (AI-assisted driver guidance)
        # ==========================================
        if missing_frames > MAX_MISSING_FRAMES:
            current_state     = "EMPTY"
            locked_truck_type = None
            driver_action     = ""
            still_frames      = 0
            out_of_zone_frames = 0
            departing_frames  = 0
            last_truck_box    = None
            type_votes        = {"BIG TRUCK": 0, "SMALL TRUCK": 0}

        else:
            if current_state == "EMPTY" and truck_found:
                current_state      = "ARRIVED"
                still_frames       = 0
                out_of_zone_frames = 0
                departing_frames   = 0

            elif current_state == "ARRIVED":
                if in_zone:
                    still_frames = still_frames + 1 if is_still else max(0, still_frames - 2)
                    driver_action = "PLEASE STOP TRUCK"
                    if still_frames >= REQUIRED_STILL_FRAMES:
                        current_state      = "LOADING"
                        out_of_zone_frames = 0
                else:
                    still_frames  = max(0, still_frames - 2)
                    
                    # --- AI guidance for driver ---
                    if target_x > (DROP_POINT_X + DROP_RADIUS):
                        driver_action = "<< PLEASE MOVE BACK"  # Not there yet, reverse further
                    elif target_x < (DROP_POINT_X - DROP_RADIUS):
                        driver_action = "MOVE FORWARD >>"      # Overshot the pipe, move forward
                    else:
                        driver_action = "ADJUST POSITION"      # In position but not yet stable

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
                driver_action    = "CLEAR TO LEAVE"
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
        if frame_count % 5 == 0:
            if truck_found:
                print(
                    f"[Frame {frame_count:04d}] "
                    f"Type: {display_truck_type:<11} | "
                    f"State: {current_state:<9} | "
                    f"Dist: {distance_to_drop:.1f}px | "
                    f"Action: {driver_action}"
                )

        # ==========================================
        # 4. Draw UI (Bullseye Circle System)
        # ==========================================
        height, width, _ = frame.shape

        # Draw plant pipe drop target
        cv2.circle(frame, (DROP_POINT_X, DROP_POINT_Y), DROP_RADIUS, (255, 0, 255), 2)
        cv2.drawMarker(frame, (DROP_POINT_X, DROP_POINT_Y), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, "DROP TARGET", (DROP_POINT_X - 50, DROP_POINT_Y - DROP_RADIUS - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if truck_found:
            if display_truck_type == "SMALL TRUCK":
                box_color = (0, 255, 0)
            elif display_truck_type == "BIG TRUCK":
                box_color = (255, 0, 0)
            else:
                box_color = (0, 255, 255)

            # Draw truck bounding box
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), box_color, 3)
            cv2.putText(frame, f"{display_truck_type}", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw swing chute box (if present and not discarded)
            if chute_box is not None:
                cx1, cy1, cx2, cy2, _ = chute_box
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
                cv2.putText(frame, "CHUTE", (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Truck target point
            cv2.circle(frame, (target_x, target_y), 6, (0, 255, 255), -1)
            
            # Draw distance line
            line_color = (0, 255, 0) if in_zone else (0, 0, 255)
            cv2.line(frame, (target_x, target_y), (DROP_POINT_X, DROP_POINT_Y), line_color, 2, cv2.LINE_AA)
            
            # Print pixel distance label on line
            mid_x = int((target_x + DROP_POINT_X) / 2)
            mid_y = int((target_y + DROP_POINT_Y) / 2)
            cv2.putText(frame, f"{int(distance_to_drop)}px", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

        # Status bar
        if current_state == "LOADING":
            ui_text, ui_color, action_color = f"READY TO LOAD [{display_truck_type}]", (255, 0, 255), (0, 255, 0)
        elif current_state == "ARRIVED":
            ui_text, ui_color, action_color = f"WAITING [{display_truck_type}]", (0, 255, 255), (0, 0, 255)
        elif current_state == "DEPARTING":
            ui_text, ui_color, action_color = f"COMPLETED [{display_truck_type}]", (255, 255, 0), (255, 255, 0)
        else:
            ui_text, ui_color, action_color = "NO TRUCK AT ZONE", (0, 0, 255), (0, 0, 255)

        cv2.putText(frame, ui_text, (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ui_color, 3)

        # Debug bar (shows truck height h for easy monitoring)
        truck_h_val = truck_h if truck_found else 0
        cv2.putText(frame,
            f"Miss:{missing_frames}/{MAX_MISSING_FRAMES} | OutZone:{out_of_zone_frames}/{REQUIRED_OUT_FRAMES} | "
            f"Still:{still_frames}/{REQUIRED_STILL_FRAMES} | Votes:{type_votes} | h:{truck_h_val}",
            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Driver action (blink only for stop warning)
        if driver_action:
            blink = int(time.time() * 2) % 2 == 0
            warning_msgs = ["PLEASE STOP TRUCK"]
            if driver_action in warning_msgs and blink:
                cv2.putText(frame, driver_action, (width // 2 - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, action_color, 4)
            elif driver_action not in warning_msgs:
                cv2.putText(frame, driver_action, (width // 2 - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, action_color, 4)

        cv2.imshow("CPAC Smart Loading Zone (Height System)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()