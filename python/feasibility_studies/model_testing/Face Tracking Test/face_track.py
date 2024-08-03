import cv2
import numpy as np

class FaceTracker:
    def __init__(self):
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialize Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.measurement = np.array((2,1), np.float32)
        self.prediction = np.zeros((2,1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.traces = []

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            self.measurement = np.array([[np.float32(x+w/2)], [np.float32(y+h/2)]])
            if len(self.traces) > 0:
                self.kalman.correct(self.measurement)
            self.prediction = self.kalman.predict()
            self.traces.append((int(self.prediction[0]), int(self.prediction[1])))

        return faces

    def draw(self, frame):
        for (x, y, w, h) in self.update(frame):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        for i in range(1, len(self.traces)):
            if self.traces[i] is not None and self.traces[i-1] is not None:
                cv2.line(frame, self.traces[i-1], self.traces[i], (0, 200, 0), 2)

        cv2.imshow('Face Tracking', frame)

def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.draw(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
