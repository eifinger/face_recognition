from os import listdir
from os.path import isfile, join, splitext

import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

import numpy as np

# Global storage for images
faces_list = []

# Create flask app
app = Flask(__name__)
CORS(app)

# <Picture functions> #


def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]


def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return list([(calc_face_encoding(image), remove_file_ext(image))
                 for image in image_files])


def detect_faces_in_image(file_stream):
    global faces_list
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)

    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)

    # Defaults for the result object
    count = len(uploaded_faces)
    faces = []

    if count:
        face_encodings = [ f[0] for f in faces_list ]
        for uploaded_face in uploaded_faces:
            face = {}
            for face_encoding in face_encodings:
                dist = face_recognition.face_distance(face_encoding,
                            uploaded_face)[0]
                name = get_name_for_face_encoding(face_encoding)
                #Check if we found a match with a lower distance (higher resemblance)
                if not "dist" in face or dist < face["dist"]:
                    face["id"] = name
                    face["dist"] = dist
            #only append if at least one face found
            if "id" in face:
                faces.append(face)
    return {
        "count": count,
        "faces": faces
    }

def get_name_for_face_encoding(face_encoding):
    global faces_list
    for entry in faces_list:
        if entry[0] == face_encoding:
            return entry[1]

# <Picture functions> #

# <Controller>


@app.route('/', methods=['POST'])
def web_recognize():
    file = extract_image(request)

    if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
        return jsonify(detect_faces_in_image(file))
    else:
        raise BadRequest("Given file is invalid!")


@app.route('/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    global faces_list
    # GET
    if request.method == 'GET':
        return jsonify(list(set([ f[1] for f in faces_list ])))

    # POST/DELETE
    file = extract_image(request)
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'POST':
        try:
            new_encoding = calc_face_encoding(file)
            faces_list.append([new_encoding, request.args.get('id')])
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == 'DELETE':
        for entry in faces_list:
            if entry[1] == request.args.get('id'):
                faces_list.remove(entry)

    return jsonify(list(set([ f[1] for f in faces_list ])))


def extract_image(request):
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file
# </Controller>


if __name__ == "__main__":
    print("Starting by generating encodings for found images...")
    # Calculate known faces
    faces_dict = get_faces_dict("/root/faces")

    # Start app
    print("Starting WebServer...")
    app.run(host='0.0.0.0', port=8080, debug=False)
