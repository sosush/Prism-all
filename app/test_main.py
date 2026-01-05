import cv2
from main import PrismEngine
import time

# Initialize Engine
engine = PrismEngine()

# Open Webcam
cap = cv2.VideoCapture(0)

print("Starting Prism Engine Test... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # FAKE DATA SIMULATION (Since we don't have Jai's ROI cropper yet)
    # We just use the center of the screen as "forehead"
    h, w, _ = frame.shape
    forehead = frame[h//3:h//2, w//3:2*w//3] # Rough center crop
    
    # Run Engine
    # We simulate screen_color as RED for testing
    result = engine.process_frame(forehead, frame, "RED")

    # Display Stats on Screen
    cv2.putText(frame, f"BPM: {result.bpm} (Q:{result.signal_quality:.2f})", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"HRV Entropy: {result.hrv_score:.3f}", (30, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(frame, f"Human: {result.is_human} ({result.confidence}%)", (30, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"SSS: {result.details.get('sss_ratio', 0):.3f} | Var: {result.details.get('signal_variance', 0):.2f}%", (30, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Prism Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()