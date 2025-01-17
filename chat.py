import customtkinter as ctk
from tkinter import messagebox
from tkinter import filedialog
import pygame
import os
import torch
from model import NeuralNetwork
from N_UTILLS import bag_of_words, tokenize, stem
import json
import random

used_tags = []

Happiness = 5
sadness = 0 
angry = 0

pygame.mixer.init()

class ChatApp:

    def __init__(self, root):
        
        self.root = root
        self.root.title("Emile & Edward(V1.2.5)")
        ctk.set_appearance_mode("dark")  # Set the appearance mode of customtkinter
        ctk.set_default_color_theme("dark-blue")  # Set the theme for the app
        self.root.geometry("800x600")
        # Load model
        self.load_model()
        self.root.after(100, self.show_startup_message)
        Happiness = 5
        sadness = 0
        angry = 0

        # Set up the UI components
        self.text_area = ctk.CTkTextbox(root, state='disabled', wrap='word', bg_color='#1c1c1c', text_color='#B57EDC')
        self.text_area.pack(padx=10, pady=10, fill='both', expand=True)

        self.entry = ctk.CTkEntry(root, fg_color='#000000', text_color='#B57EDC')
        self.entry.pack(padx=10, pady=10, fill='x')

        self.send_button = ctk.CTkButton(root, text="Send", command=self.get_response, fg_color='#8A2BE2', text_color='#ffffff')
        self.send_button.pack(padx=10, pady=10)

        
        


    def play_sound(self):
    
        file_path = os.path.join(os.getcwd(), "sound1.wav")
        if file_path:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()


    def show_startup_message(self):
        # Create a custom pop-up window
        popup = ctk.CTkToplevel(self.root)
        popup.title("Welcome")

        # Set the message and layout
        message = ("Welcome to Emile & Edward(V1.2.5)\n"
                   "Consider this chatbot has a small dataset\n"
                   "and isn't as advanced as other chatbots\n"
                   "ike ChatGPT and Claude.")
        label = ctk.CTkLabel(popup, text=message, padx=20, pady=20)
        label.pack()

        ok_button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        ok_button.pack(pady=10)

        # Center the pop-up window
        

        # Make the pop-up window always on top
        popup.attributes('-topmost', True)
        popup.attributes('-topmost', False)

        self.center_window(popup, 300, 150)

    def center_window(self, window, width, height):
        # Get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Calculate position for centering the window
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Set the window size and position
        window.geometry(f"{width}x{height}+{x}+{y}")

    


    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load chatbot data and model
        with open("EMILE_DATA.json", 'r') as f:
            self.emile_data = json.load(f)
        
        data = torch.load("best_model.pth")
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        model_state = data["model_state"]

        self.model = NeuralNetwork(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def get_response(self):
        user_input = self.entry.get()
        self.text_area.configure(state='normal')
        self.play_sound()
        self.text_area.insert(ctk.END, "You: " + user_input + "\n")

        
        sentence = tokenize(user_input)
        
        Amount = []
        
        Last_num = 0
        for i,w in enumerate(sentence):
            w = stem(w)
            if w == "and" and i != 0:
                print("multi-question") 
                Amount.append(i)
            Last_num = i
        
        Last_num = Last_num + 1
        print(Last_num)
        
        print(sentence)
        print(Amount)

        Response = []
        Round = 0
        if len(Amount) > 0 :
            Amount.append(Last_num)
            for e in range(len(Amount)):
                print(Amount[e])
                sentence2 = []
                for i in range(Round,Amount[e]):
                    
                    if(sentence[i] != "and"):
                        sentence2.append(sentence[i])
                        print(sentence[i])
                    

                Xx = bag_of_words(sentence2, self.all_words)
                Xx = torch.from_numpy(Xx).float().unsqueeze(0).to(self.device)
                print("PASSED")
                Round = Amount[e]
                Response.append(self.answerr(Xx,sentence))
                print(Response)
        else :

            Xx = bag_of_words(sentence, self.all_words)
            Xx = torch.from_numpy(Xx).float().unsqueeze(0).to(self.device)
            Response.append(self.answerr(Xx,sentence))
        
        Re = ""
        for Resp in Response :
            Re = Re + Resp

        used_tags.clear()
        self.Responed(Re)
        
        
        

        

        

    def answerr(self,X,Sentence_):
        with torch.no_grad():
            output = self.model(X)
            prop = torch.softmax(output, dim=1)
            probs, predicted = torch.max(prop, dim=1)
            tag = self.tags[predicted.item()]
            confidence = probs.item()
            
            
            if confidence > 0.6:
                response = "Sorry, I don't have a response."  # Default response
                for intent in self.emile_data["intents"]:
                    if tag == intent['tag']:
                        valid = True

                        


                        for ut in used_tags:
                            if ut == tag :
                                valid = False
                            
                        
                        
                        if valid == True :
                            response = random.choice(intent["responses"])
                            used_tags.append(tag)
                            if tag == "feeling" :
                                if Happiness > sadness and Happiness > angry:
                                    return ("I'm feeling happy, thanks cutie! What about you?" + " ")
                                elif sadness > Happiness and sadness > angry : 
                                    return ("I'm feeling okay, I'm simply feeling a little sad, thanks cutie" + " ")
                                elif angry > Happiness and angry > sadness : 
                                    return ("Telling my feelings to you is a major mistake." + " ")
                            elif tag == "cal" :
                                responses = ""
                                responses = str(self.Calculations(Sentence_))

                                return ("The answer to your math problem is :  " + responses)
                            else :
                                return (response + " ")
                        else :
                            response = ""
                            return response
                        
            else:
                response = "Sorry, I don't understand."
                return response

    def Responed(self,Res):
        self.text_area.insert(ctk.END," " + "\n")
        self.text_area.insert(ctk.END, "\nEmilie: " + Res + "\n")
        self.text_area.configure(state='disabled')
        self.entry.delete(0, ctk.END)
    
    def Feelings(self, tag,happinesss, sadnesss, angrys):
        print("working")
        if tag == "greeting" or tag == "thanking" or tag == "sayingf" or  tag == "makh" or tag == "happiness" or tag == "comfort": 
            happinesss += 1
            sadnesss -= 1
            return happinesss, sadnesss, angrys
        elif tag == "sayingfs":
            sadnesss += 1
            happinesss -= 1
            return happinesss, sadnesss, angrys
        elif tag == "sarcasm" :
            angrys += 1
            happiness -= 1
            return happinesss, sadnesss, angrys

    def Calculations(self,words):
        Answer = 0
        Multi = []
        divide = []
        plus = []
        minus = []

        Taken_Matrix = []

        print(words)
        for i, Word in enumerate(words):
            print(Word)

            
            if Word == "*" or Word == "multipy" or Word == "mult":
                #Answer = Answer + (float(words[i - 1]) * float(words[i + 1])) 
                valid = False
                print(len(words))
                print((i + 2))

                
                if len(words) > ((i + 2) + 1) :
                    if words[i + 2] != "+" and words[i + 2] != "-" :
                        Multi.append(float(words[i + 1])) 
                        Taken_Matrix.append(i + 1)
                        print("append" + words[i- 1] )
                else :
                    Multi.append(float(words[i + 1])) 
                    valid = True

                print("wtf" + words[i- 1] )

                for CO in Taken_Matrix :
                    print("Checking Taken Matrix: [I - 1] : ",i - 1, " MATRIX : ", CO )
                    if i - 1 == CO :
                        valid = True
                        print("FOUND IT")
                        
                if words[i - 2] != "+" and words[i - 2] != "-" and valid == False:
                    Multi.append(float(words[i - 1]))
                    Taken_Matrix.append(i - 1)
                    print("append" + words[i- 1] )
                
            elif Word == "/" or Word == "divide" or Word == "div" :
                #Answer = Answer + (float(words[i - 1]) / float(words[i + 1])) 
                divide.append(float(words[i - 1]))
                divide.append(float(words[i + 1]))
            elif Word == "+" or Word == "plus" or Word == "pl" :
                #Answer = Answer + (float(words[i - 1]) + float(words[i + 1])) 

                valid = False
                
                for CO in Taken_Matrix :
                    print("Checking Taken Matrix: [I - 1] : ",i - 1, " MATRIX : ", CO )
                    if i - 1 == CO :
                        valid = True
                        print("FOUND IT")

                if valid == False :
                    plus.append(float(words[i - 1]))
                    Taken_Matrix.append(i - 1)
                    print("append" + words[i- 1] )
                    

                print(len(words))
                if len(words) >= ((i + 1) + 2):
                    if valid == False : 
                        plus.append(float(words[i + 1]))
                        Taken_Matrix.append(i + 1)
                        print("append" + words[i + 1] )
                else :
                    plus.append(float(words[i + 1]))
                

                print("plus")
            elif Word == "-" or Word == "minus" or Word == "mi" :
                #Answer = Answer + (float(words[i - 1]) - float(words[i + 1])) 
                minus.append(float(words[i - 1]))
                minus.append(float(words[i + 1]))
                Taken.append(i)
            

        multi_ = 1
        divide_ = 1
        plus_ = 0
        minus_ = 0

        print("Matrix  :  ",Taken_Matrix)

        for i in range(len(Multi)):
            print(i)
        for M in Multi :
            multi_ = M * multi_
            print("Multipie values : ", multi_)

        for D in divide :
            divide_ = D / divide_

        for P in plus :
            plus_ = P + plus_

        for MM in minus :
            minus_ = MM - minus_

        if len(divide) <= 0 :
            divide_ = 0

        if len(Multi) <= 0 :
            multi_ = 0

        Answer = multi_ + divide_ + plus_ + minus_
        
        return Answer
                
        #divitions : Word == "divide" or Word == "/"



if __name__ == "__main__":
    root = ctk.CTk()
    app = ChatApp(root)
    root.mainloop()




