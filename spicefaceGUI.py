import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
#import dlib.cuda as cuda
import cv2
import numpy as np
import keyboard
import datetime


import time

import Tkinter as tk
from PIL import Image, ImageTk

import os

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)

gc = gspread.authorize(credentials)


#print("{}: {}".format("get_num_devices", cuda.get_num_devices()))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


cam = cv2.VideoCapture(0)

chatON = False

if chatON:
    client = Client('login@gmail.com', 'password')

if chatON:
    thread_id = "2589250877804146"
    thread_type = ThreadType.GROUP

# Will send a message to the thread
if chatON:
    client.send(Message(text="Witaj swiecie. Jestem Billy. Nie chce przejac kontroli nad swiatem, a pozatym sie uruchomilem w warsztacie"), thread_id=thread_id, thread_type=thread_type)
    client.changeThreadColor(ThreadColor.BILOBA_FLOWER, thread_id=thread_id)


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.52):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def predict_img(X_img, knn_clf=None, model_path=None, distance_threshold=0.52):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # find face locations
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("dataset", model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")
    
    #otwieranie arkuszy
    now = datetime.datetime.now()
    
    wks_miesiac = gc.open("Obecnosci2019").worksheet(now.strftime("%Y.%m"))
    wks_podsumowanie = gc.open("Obecnosci2019").worksheet("Podsumowanie")
    
    # # # GUI
    def show_entry_fields():
        print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
    
    master = tk.Tk()
    master.wm_title("SpiceFace")
    #master.config(background="#FFFFFF")
    master.attributes("-fullscreen", True)
    
    tk.Label(master, 
             text="Wpisz swoje imie i nazwisko (bez polskich znaków). Spojrz w kamerę i kliknij HOME. Poczekaj na napis.",
             font = "Helvetica 20 bold italic").grid(row=0, columnspan=2)
    tk.Label(master, 
             text="First Name").grid(row=1), 
    tk.Label(master, 
             text="Last Name").grid(row=2)

    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    


    e1.grid(row=1, column=1)
    e2.grid(row=2, column=1)
                                        
    # tk.Button(master, 
    #           text='Dodaj', command=show_entry_fields).grid(row=3, 
    #                                                        column=1, 
    #                                                        sticky=tk.W, 
    #                                                        pady=4)
                                                           
    imageFrame = tk.Frame(master, width=640, height=480)
    imageFrame.place(relx=0.5, rely=0.5, anchor= tk.CENTER)
    #Capture video frames
    lmain = tk.Label(imageFrame)
    lmain.grid(row=0, column=0)
    
    faceStartT = 0
    isStillFace = False
    faces_len = 0
    
    def show_frame():
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  
        global faces_len
        
        faces_len = len(faces)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x-25,y-25),(x+w+50,y+h+50),(128,0,0),2)
            roi_gray = frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
    
    tk.Label(master, 
        text="Kliknij spacje lub dodaj sie jako nowy użytkownik",
        fg = "black",
        #bg = "dark green",
        compound = tk.CENTER,
        font = "Helvetica 20 bold italic").place(relx=0.5, rely=0.2, anchor= tk.CENTER)
        
    infoText = tk.StringVar()
    
    tk.Label(master, 
        textvariable = infoText,
        fg = "black",
        #bg = "dark green",
        compound = tk.CENTER,
        font = "Helvetica 20 bold italic").place(relx=0.5, rely=0.8, anchor= tk.CENTER)
        
    listbox = tk.Listbox(master)
    
    listbox.place(relx=0.8, rely=0.5, anchor= tk.CENTER)
    
    cell_list = wks_podsumowanie.findall("in")
    i = 1
    for cell in cell_list:
        i_row = cell.row
        imie = wks_podsumowanie.cell(i_row, 1).value
        listbox.insert(i, imie)
        i += 1
        
    def DeleteSelection() :
        # items = self.list_box_1.curselection()
        nameOfPerson = listbox.get(tk.ANCHOR)
        name_row = wks_podsumowanie.find(nameOfPerson).row
        name_status = wks_podsumowanie.cell(name_row, 2).value

        wks_miesiac.update_cell(name_status, 3, now.strftime("%Y-%m-%d %H:%M:%S"))
        wks_podsumowanie.update_cell(name_row, 2, "out")
        #print("{} wychodzi z warsztatu".format(name))
        infoText.set("{} wychodzi z warsztatu".format(nameOfPerson))
        client.send(Message(text="{} wychodzi do warsztatu".format(nameOfPerson)), thread_id=thread_id, thread_type=thread_type)

        #print nameOfPerson
        listbox.delete(tk.ANCHOR)

    
    b = tk.Button(master, text="Delete", command=DeleteSelection)
    b.place(relx=0.8, rely=0.65, anchor= tk.CENTER)
    
    show_frame()  #Display 2

    countA = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, None, fx=0.38, fy=0.38, interpolation=cv2.INTER_AREA)

        now = datetime.datetime.now()

        countA = countA + 1

        if(countA > 2000):
            gc = gspread.authorize(credentials)
            wks_miesiac = gc.open("Obecnosci2019").worksheet(now.strftime("%Y.%m"))
            wks_podsumowanie = gc.open("Obecnosci2019").worksheet("Podsumowanie")
            countA = 0
        
        # global faceStartT
        # global isStillFace
        rejstracja = False
        
        if(faces_len == 1):
            dtx = datetime.datetime.now()
            
            if(isStillFace == False):
                faceStartT = dtx.second
                isStillFace = True
                rejstracja = False
            else:
                timeofFace = dtx.second - faceStartT
                if(timeofFace > 1):
                    faceStartT = dtx.second
                    isStillFace = False
                    rejstracja = False
                    #print("Rejstracja")
        else:
            isStillFace = False
            rejstracja = False
            # a wciskami gdy wchodzimy do warsztatu

        # if  
        if(rejstracja) or keyboard.is_pressed('space'):
           # robienie predykcji
            isStillFace = False
            print("spacja")

            infoText.set("Analizuje... poczekaj chwilkę")

            predictions = predict_img(frame, model_path="trained_knn_model.clf")
            
            if len(predictions) == 1:
                for name, (top, right, bottom, left) in predictions:
                    name = name.encode("UTF-8")
                    #print(name)
                    if name != "unknown":
                        #zapisywanie zdjęcia do folderu osoby
                        outfile = 'dataset/{}/{}_{}.jpg'.format(name, name, now.strftime("%Y-%m-%d-%H-%M-%S-%f"))
                        cv2.imwrite(outfile, frame)
                                                
                        # get status of name.
                        # number is working row and person is in workshop 
                        name_row = wks_podsumowanie.find(name).row
                        name_status = wks_podsumowanie.cell(name_row, 2).value
                        
                        if name_status == "out":
                            #zapisywanie godziny wejścia osoby
                            cell = wks_miesiac.find("o")
                            row_number = cell.row
                            column_number = cell.col
                            wks_miesiac.update_cell(row_number, 1, name)
                            wks_miesiac.update_cell(row_number, 2, now.strftime("%Y-%m-%d %H:%M:%S"))
                            wks_podsumowanie.update_cell(name_row, 2, row_number)
                            if chatON:
        			    client.send(Message(text="{} wchodzi do warsztatu".format(name)), thread_id=thread_id, thread_type=thread_type)
                            print("{} wchodzi do warsztatu".format(name))
                            infoText.set("{} wchodzi do warsztatu".format(name))


                        else:
                            wks_miesiac.update_cell(name_status, 3, now.strftime("%Y-%m-%d %H:%M:%S"))
                            wks_podsumowanie.update_cell(name_row, 2, "out")
                            print("{} wychodzi z warsztatu".format(name))
                            infoText.set("{} wychodzi z warsztatu".format(name))
                            if chatON:
                                client.send(Message(text="{} wychodzi do warsztatu".format(name)), thread_id=thread_id, thread_type=thread_type)
                            isclose = wks_podsumowanie.cell(1, 2).value

                            if chatON:
                                if isclose == "close":
                                    client.send(Message(text="Wszyscy wszyli z warsztatu. :("), thread_id=thread_id, thread_type=thread_type)
                                    client.changeThreadColor(ThreadColor.MESSENGER_BLUE, thread_id=thread_id)


                    else:
                        #zapisywanie zdjęcia do folderu osoby
                        outfile = 'dataset/unknown/{}.jpg'.format(now.strftime("%Y-%m-%d-%H-%M-%S-%f"))
                        cv2.imwrite(outfile, frame)

                        infoText.set("Nie poznaje twojej twarzy. Jeśli jesteś nowy dodaj sie w lewym górnym rogu")
                
                listbox.delete(0,tk.END)
                cell_list = wks_podsumowanie.findall("in")
                i = 1
                for cell in cell_list:
                    i_row = cell.row
                    imie = wks_podsumowanie.cell(i_row, 1).value
                    listbox.insert(i, imie)
                    i += 1
            elif len(predictions) > 1:
                infoText.set("Za dużo twarzy!")
                
            else:
                infoText.set("Nie znaleziono twarzy. Spróbuj jeszcze raz.")
                        
        if keyboard.is_pressed('home'):
            # Adding new person to database
            print("home")

            ###GETTING NAME AND SECOUND NAME
            first_name = e1.get()
            second_name = e2.get()
            
            infoText.set("Analizuje... {}_{}. Patrz w kamera".format(first_name, second_name))
            print("Dodawanie nowej osoby... {}_{}".format(first_name, second_name))
            
            new_id = "{}_{}".format(first_name, second_name)
            print(new_id)
            
            # Creating folder for new person
            if not os.path.exists("dataset/{}".format(new_id)):
                
                os.makedirs("dataset/{}".format(new_id))
            
                outfile = 'dataset/{}/{}_{}.jpg'.format(new_id, new_id, now.strftime("%Y-%m-%d-%H-%M-%S-%f"))
                
                cv2.imwrite(outfile, frame)
                
                for x in [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0]:
                    now = datetime.datetime.now()
                    infoText.set("Patrz w kamere przez {}s".format(x))
                    ret, frame = cam.read()
                    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    time.sleep(0.5)
                    outfile = 'dataset/{}/{}_{}.jpg'.format(new_id, new_id, now.strftime("%Y-%m-%d-%H-%M-%S-%f"))
                    cv2.imwrite(outfile, frame)
                    master.update()

                infoText.set("{} w przeciągu 24 godzin zostaniesz dodany do bazy danych. Witamy w Spice Gears".format(new_id))

                # Adding new person to gspread
                name_row = wks_podsumowanie.find("-").row
                wks_podsumowanie.update_cell(name_row, 1, new_id)
            else:
                infoText.set("{} już istnieje".format(new_id))
        # else:
        #     print("inny klawisz")
                    
       # except:      
         #   print("inny klawisz")
            

        master.update()

# Release handle to the webcam
cam.release()
cv2.destroyAllWindows()
