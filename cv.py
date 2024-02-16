# import numpy as np
# import cv2
# import time

# # initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cv2.startWindowThread()

# # open webcam video stream
# cap = cv2.VideoCapture(0)

# # the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640,480))

# # initialize dictionary to store start times for each person
# start_times = {}

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
#     # using a greyscale picture, also for faster detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

#     # detect people in the image
#     # returns the bounding boxes for the detected objects
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

#     for (xA, yA, xB, yB) in boxes:
#         # display the detected boxes in the colour picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB),
#                           (0, 255, 0), 2)
        
#         # check if a person is already being tracked
#         person_key = None
#         for key, bbox in start_times.items():
#             if xA >= bbox[0] and yA >= bbox[1] and xB <= bbox[2] and yB <= bbox[3]:
#                 person_key = key
#                 break
        
#         if person_key is None:
#             # Start timer if a new person is detected
#             start_times[len(start_times)] = (xA, yA, xB, yB, time.time())
#         else:
#             # If person is already being tracked, update the bounding box
#             start_times[person_key] = (xA, yA, xB, yB, start_times[person_key][4])

#             # Draw timer text on the head of detected person
#             current_time = time.time()
#             elapsed_time = current_time - start_times[person_key][4]
#             hours = int(elapsed_time // 3600)
#             minutes = int((elapsed_time % 3600) // 60)
#             seconds = int(elapsed_time % 60)
#             timer_text = f'Time: {hours:02d}:{minutes:02d}:{seconds:02d}'
#             cv2.putText(frame, timer_text, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Check for inactive timers
#     inactive_timers = []
#     for key in list(start_times.keys()):  # Create a copy of the keys
#         bbox = start_times[key]
#         current_time = time.time()
#         elapsed_time = current_time - bbox[4]
#         if elapsed_time > 5:  # If person is not visible for more than 5 seconds
#             inactive_timers.append((key, elapsed_time))
#             del start_times[key]
    
#     # Write the output video 
#     out.write(frame.astype('uint8'))
#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# # and release the output
# out.release()
# # finally, close the window
# cv2.destroyAllWindows()
# cv2.waitKey(1)

# # Print inactive timers
# for timer in inactive_timers:
#     print(f"Person {timer[0]} was inactive for {timer[1]} seconds.")


# import numpy as np
# import cv2
# import time

# # Initialize the HOG descriptor/person detector for customers
# hog_customer = cv2.HOGDescriptor()
# hog_customer.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# # Initialize the face detector using dnn module
# net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# # Load the employee's facial photo for recognition
# employee_photo = cv2.imread("C:/Users/HARGUN/Desktop/HARGUN/Hargun Profile Pic.jpg")
# employee_photo = cv2.resize(employee_photo, (150, 150))

# cv2.startWindowThread()

# # Open webcam video stream
# cap = cv2.VideoCapture(0)

# # The output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640, 480))

# # Initialize dictionary to store start times for each person (customers and employees)
# start_times_customers = {}
# start_times_employees = {}

# # Initialize dictionary to store activity of employees (e.g., number of glasses delivered)
# employee_activity = {}

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))

#     # Detect customers in the image
#     boxes_customers, weights_customers = hog_customer.detectMultiScale(frame, winStride=(8,8))
#     boxes_customers = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_customers])

#     for (xA, yA, xB, yB) in boxes_customers:
#         # Display the detected boxes for customers in the color picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
#         # Check if a customer is already being tracked
#         if start_times_customers.get((xA, yA, xB, yB)) is None:
#             # Start timer if a new customer is detected
#             start_times_customers[(xA, yA, xB, yB)] = time.time()
#         else:
#             # Calculate time if customer is already being tracked
#             current_time = time.time()
#             elapsed_time = current_time - start_times_customers[(xA, yA, xB, yB)]
#             hours = int(elapsed_time // 3600)
#             minutes = int((elapsed_time % 3600) // 60)
#             seconds = int(elapsed_time % 60)
#             timer_text = f'Customer Time: {hours:02d}:{minutes:02d}:{seconds:02d}'
#             cv2.putText(frame, timer_text, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Detect faces in the image
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         # Filter out weak detections
#         if confidence > 0.5:
#             # Get the bounding box coordinates
#             box = detections[0, 0, i, 3:7] * np.array([640, 480, 640, 480])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Extract the face ROI
#             face = frame[startY:endY, startX:endX]
#             (fH, fW) = face.shape[:2]

#             # Ensure the face width and height are sufficiently large
#             if fW < 20 or fH < 20:
#                 continue

#             # Perform face recognition
#             # Here you would compare the detected face with the employee photo using face recognition algorithms
#             # For demonstration purposes, let's just draw a rectangle around the detected face
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

#             # Display name above the detected face
#             name_text = "Hargun Singh Chandhok"  # Replace this with the detected person's name
#             text_width, text_height = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
#             text_x = max(0, startX + int((endX - startX - text_width) / 2))
#             text_y = max(0, startY - 10)
#             cv2.putText(frame, name_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # Write the output video 
#     out.write(frame.astype('uint8'))
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# cap.release()
# # Release the output
# out.release()
# # Close the window
# cv2.destroyAllWindows()
# cv2.waitKey(1)



import numpy as np
import cv2
import time

# Initialize the HOG descriptor/person detector for customers
hog_customer = cv2.HOGDescriptor()
hog_customer.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the face detector using dnn module
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load the employee's facial photo for recognition
employee_photo = cv2.imread("C:/Users/HARGUN/Desktop/HARGUN/Hargun Profile Pic.jpg")
employee_photo = cv2.resize(employee_photo, (150, 150))

cv2.startWindowThread()

# Open webcam video stream
cap = cv2.VideoCapture(0)

# The output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

# Initialize dictionary to store start times for each person (customers and employees)
start_times_customers = {}
start_times_employees = {}

# Initialize dictionary to store activity of employees (e.g., number of glasses delivered)
employee_activity = {}

# Initialize dictionary to track interactions between employees and customers
interactions = {}

# Set the proximity threshold for interaction
proximity_threshold = 50  # Adjust as needed

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # Detect customers in the image
    boxes_customers, weights_customers = hog_customer.detectMultiScale(frame, winStride=(8,8))
    boxes_customers = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_customers])

    for (xA, yA, xB, yB) in boxes_customers:
        # Display the detected boxes for customers in the color picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        # Check if a customer is already being tracked
        if start_times_customers.get((xA, yA, xB, yB)) is None:
            # Start timer if a new customer is detected
            start_times_customers[(xA, yA, xB, yB)] = time.time()
        else:
            # Calculate time if customer is already being tracked
            current_time = time.time()
            elapsed_time = current_time - start_times_customers[(xA, yA, xB, yB)]
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            timer_text = f'Customer Time: {hours:02d}:{minutes:02d}:{seconds:02d}'
            cv2.putText(frame, timer_text, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect faces in the image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([640, 480, 640, 480])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Perform face recognition
            # Here you would compare the detected face with the employee photo using face recognition algorithms
            # For demonstration purposes, let's just draw a rectangle around the detected face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Display name above the detected face
            name_text = "Hargun Singh Chandhok"  # Replace this with the detected person's name
            text_width, text_height = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = max(0, startX + int((endX - startX - text_width) / 2))
            text_y = max(0, startY - 10)
            cv2.putText(frame, name_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check for interactions between employees and customers
            for customer_bbox in boxes_customers:
                cx, cy, cxb, cyb = customer_bbox
                cx_center, cy_center = (cx + cxb) // 2, (cy + cyb) // 2
                if abs((startX + endX) // 2 - cx_center) <= proximity_threshold and abs((startY + endY) // 2 - cy_center) <= proximity_threshold:
                    # Interaction detected between employee and customer
                    employee_activity[name_text] = employee_activity.get(name_text, 0) + 1
                    # Record the interaction to avoid duplicate counts
                    interactions[(startX, startY, endX, endY, cx, cy, cxb, cyb)] = time.time()
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
# Release the output
out.release()
# Close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
