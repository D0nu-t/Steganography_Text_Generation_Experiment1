# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:40:01 2020

@author: Krishnasai Addala
"""
print("test")
import numpy
import nltk
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import PIL
from PIL import Image
nltk.download('stopwords')

file = open("ref2.txt","r",encoding="utf8")
rawt=file.read()

def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)


processed_inputs = tokenize_words(rawt)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

seq_length = 100
x_data = []
y_data = []

for i in range(0, input_len - seq_length, 1):
    in_seq = processed_inputs[i:i + seq_length]

    out_seq = processed_inputs[i + seq_length]

    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])



n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


filename = "model_weights_saved.hdf5"
model.load_weights(filename)

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)

# num_to_char = dict((i, c) for i, c in enumerate(chars))
# res=""
# start = numpy.random.randint(0, len(x_data) - 1)
# pattern = x_data[start]
# print("Random Seed:")
# print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

# for i in range(50):
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(vocab_len)
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = num_to_char[index]
#     seq_in = [num_to_char[value] for value in pattern]

#     sys.stdout.write(result)

#     pattern.append(index)
#     pattern = pattern[1:len(pattern)]
#     res+=result
    
print("\n\ntest")
    
    

def genData(data):
 
        newd = []
 
        for i in data:
            newd.append(format(ord(i), '08b'))
        return newd
 

def modPix(pix, data):
 
    datalist = genData(data)
    lendata = len(datalist)
    imdata = iter(pix)
 
    for i in range(lendata):
 
        pix = [value for value in imdata.__next__()[:3] +
                                imdata.__next__()[:3] +
                                imdata.__next__()[:3]]
 
       
        for j in range(0, 8):
            if (datalist[i][j] == '0' and pix[j]% 2 != 0):
                pix[j] -= 1
 
            elif (datalist[i][j] == '1' and pix[j] % 2 == 0):
                if(pix[j] != 0):
                    pix[j] -= 1
                else:
                    pix[j] += 1
              
 
       
        if (i == lendata - 1):
            if (pix[-1] % 2 == 0):
                if(pix[-1] != 0):
                    pix[-1] -= 1
                else:
                    pix[-1] += 1
 
        else:
            if (pix[-1] % 2 != 0):
                pix[-1] -= 1
 
        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]    
        
        
def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)
 
    for pixel in modPix(newimg.getdata(), data):
 
        newimg.putpixel((x, y), pixel)
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1
            
            
def encode():
    img = input("Enter image name(with extension) : ")
    image = Image.open(img, 'r')
    o=int(input("to enter your own data for encoding enter 1\n\n to use computer generated text,enter 2\n\n--:"))
    if(o==1):
        data = input("Enter data to be encoded : ")
        newimg = image.copy()
        encode_enc(newimg, data)
        new_img_name = input("Enter the name of new image(with extension) : ")
        newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))
        if (len(data) == 0):
                raise ValueError('Data is empty')
    elif(o==2):
        num_to_char = dict((i, c) for i, c in enumerate(chars))
        res=""
        start = numpy.random.randint(0, len(x_data) - 1)
        pattern = x_data[start]
        print("Random Seed:")
        print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

        for i in range(50):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(vocab_len)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = num_to_char[index]
            seq_in = [num_to_char[value] for value in pattern]
        
            sys.stdout.write(result)
        
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            res+=result
        
        data=res
        print(data)
        k=int(input(("this is the data to be encoded\nto generate something new, enter 1 else 2\n-:")))
        if(k==1):
            print("please wait a moment")
            for i in range(50):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(vocab_len)
                prediction = model.predict(x, verbose=0)
                index = numpy.argmax(prediction)
                result = num_to_char[index]
                seq_in = [num_to_char[value] for value in pattern]

                sys.stdout.write(result)

                pattern.append(index)
                pattern = pattern[1:len(pattern)]
                res=result
                
            data=res
            newimg = image.copy()
            encode_enc(newimg, data)
            new_img_name = input("Enter the name of new image(with extension) : ")
            newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))
            
            
        else:
            newimg = image.copy()
            encode_enc(newimg, data)
            new_img_name = input("Enter the name of new image(with extension) : ")
            newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))
            
            
        # newimg = image.copy()
        # encode_enc(newimg, data)
        # new_img_name = input("Enter the name of new image(with extension) : ")
        # newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))
    else:
        raise Exception("Enter correct input")
        
    
    
   

def decode():
    img = input("Enter image name(with extension) : ")
    image = Image.open(img, 'r')
 
    data = ''
    imgdata = iter(image.getdata())
 
    while (True):
        pixels = [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]]
 
        binstr = ''
 
        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'
 
        data += chr(int(binstr, 2))
        if (pixels[-1] % 2 != 0):
            return data


def main():
    a = int(input(":: Welcome to [placeholder] ::\n"
                        "1. Encode\n2. Decode\n3.train model\n4.Exit\n\n\n--:"))
    if (a == 1):
        encode()
        main()
 
    elif (a == 2):
        print("Decoded Word :  " + decode())
        main()
    elif(a==3):
        model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)
        main()
        
    elif(a==4):
        print("exiting program")
        return()
    else:
        raise Exception("Enter correct input")
 
if __name__ == '__main__' :
 
    main()
                    
print("\n\ntest")
    