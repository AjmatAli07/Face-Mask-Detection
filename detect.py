import cv2
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Image preprocessing
transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])


# CNN Model
class MaskDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                3,16,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,32,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,64,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64*16*16,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self,x):

        x=self.conv(x)
        x=self.fc(x)

        return x


# Load trained model
model=MaskDetector().to(device)

model.load_state_dict(
    torch.load(
        "models/mask_detector.pth",
        map_location=device
    )
)

model.eval()


labels=[
    "With Mask",
    "No Mask"
]


# Face detector
face=cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Webcam
cap=cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot access webcam")
    exit()


while True:

    ret,frame=cap.read()

    if not ret:
        break


    # Improve brightness in dark rooms
    frame=cv2.convertScaleAbs(
        frame,
        alpha=1.2,
        beta=25
    )


    gray=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces=face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
    )


    for(x,y,w,h) in faces:


        img=frame[
            y:y+h,
            x:x+w
        ]


        img=Image.fromarray(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )
        )


        img=transform(img)

        img=img.unsqueeze(0).to(device)


        with torch.no_grad():

            prediction=model(img)


            probability=torch.softmax(
                prediction,
                dim=1
            )


            confidence,pred=torch.max(
                probability,
                dim=1
            )


        confidence=confidence.item()*100

        text=f"{labels[pred.item()]} {confidence:.1f}%"


        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    # ESC to exit
    if cv2.waitKey(1)==27:
        break


cap.release()

cv2.destroyAllWindows()