import tkinter as tk
from anzi import SentenceandDistractors
import re

def process_input():
    sentences = Input.get("1.0","end-1c")
    
    for text in Texts:
        text.pack_forget()
    
    for sentence in sentences.split('\n'):
        if sentence != '':
            text = tk.Text(MiddleArea, height=5, width=100)
            data = SentenceandDistractors(sentence, v='3.2')
            Sentence = data['sentence'] + '\n(A) {} (B) {} (C) {} (D) {}'.format(data['distractor'][0], data['distractor'][1], data['distractor'][2], data['answer'])
            index = re.search('\[SEP\]', Sentence).start()
            index1 = re.search('\(A\)', Sentence).start()
            print("answer:", data['answer'])
            Sentence1 = Sentence[index1:].replace('[MASK]', '____')
            Sentence = Sentence[6:index].replace('[MASK]', '____')
            print(Sentence1)
            text.insert(tk.END, Sentence+"\n")
            text.insert(tk.END, Sentence1)
            text.pack()
            Texts.append(text)
    

Texts = []

MAX_WIDTH = 700

Root = tk.Tk()

TopArea = tk.Frame(Root)
TopArea.pack()

MiddleArea = tk.Frame(Root)
MiddleArea.pack(pady=10)

Input = tk.Text(TopArea, height=10, width=100)
Input.pack(side='left', pady=20, padx=20)

Button = tk.Button(TopArea, text='Process', command=process_input)
Button.pack(side='left', padx=20)





Root.mainloop()