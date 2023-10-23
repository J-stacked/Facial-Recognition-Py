#################### imports ###################################
from pathlib import Path
import argparse
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition
import time

#################### define constants ##########################
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
FACIAL_FEATURES_COLOR = "red"

#################### define paths, create if not existing ######
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

#################### set up command line args ##################
parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Train using: \"hog\" (CPU)(default), \"cnn\" (GPU)")
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
args = parser.parse_args()

#################### function definitions #####################    
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    print("Starting encoding...")
    start_time = time.time()
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        individual_time = time.time()
        try:
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            print(filepath, "successfully encoded in", f'{time.time()-individual_time:.3f}', "seconds.")
        except MemoryError:
            print(filepath, "failed encoding due to lack of memory")
        except:
            print(filepath, "failed encoding")
        
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
            
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
    print("Encoding complete. Time: ", f'{time.time()-start_time:.3f}', "seconds.")


def _display_face(draw, bounding_box, name, face_landmarks):
    #### draw bounding box around face
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline = BOUNDING_BOX_COLOR, width = 2)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill = BOUNDING_BOX_COLOR, outline = BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), name, fill= TEXT_COLOR)

    #### draw facial features out
    for facial_feature in face_landmarks.keys():
        draw.line(face_landmarks[facial_feature], width=3, fill = FACIAL_FEATURES_COLOR)
    

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    try:
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
    except FileNotFoundError:
        print("Encoding file not found.  Please encode known faces first.")
        return None

    try:
        input_image = face_recognition.load_image_file(image_location)
    except FileNotFoundError:
        print("Input image not found.  Please try again.")
        return None
    
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    face_landmarks_list = face_recognition.face_landmarks(input_image)
    
    pillow_image = Image.fromarray(input_image)
    drawObj = ImageDraw.Draw(pillow_image)
    
    for bounding_box, unknown_encoding, face_landmarks in zip(input_face_locations, input_face_encodings, face_landmarks_list):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name: 
            name = "Unknown"
        #print(name, bounding_box)
        _display_face(drawObj, bounding_box, name, face_landmarks)

    del drawObj
    pillow_image.show()


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
