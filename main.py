from PIL import Image
import hashlib, os, math, time



class VectorCompare:
    def magnitude(self, concordance):
        total = 0
        #print "DADA"
        for word, count in concordance.iteritems():
            total += count**2
        return math.sqrt(total)
    def relation(self, concordance1, concordance2):
        relevance = 0
        #print "mahla"
        topvalue = 0
        for word, count in concordance1.iteritems():
            if concordance2.has_key(word):
                topvalue += count * concordance2[word]
        return topvalue/(self.magnitude(concordance1) * self.magnitude(concordance2))




   # return topvalue / (magnitude(concordance1) * magnitude(concordance2))

'''
def __init__(captcha=""):
    """
    Initialize main CIntruder
    """
    captcha = set_captcha(captcha)
    start = time.time()
'''




def buildvector( im):
    d1 = {}
    count = 0
    for i in im.getdata():
        d1[count] = i
        count += 1
    return d1

def crack():
    v = VectorCompare()
    iconset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    imageset = []
    last_letter = None
    print "Loading dictionary... "
    for letter in iconset:
        for img in os.listdir('iconset/%s/'%(letter)):
            temp = []
            if img != "Thumbs.db": # win32 check
            
                if last_letter != letter:
                    print "-----------------"
                    print "Word:", letter 
                    print "-----------------"
                print img
                last_letter = letter           
                temp.append(buildvector(Image.open("./iconset/%s/%s"%(letter,img))))
            imageset.append({letter:temp})


            
    correctcount = 0
    wrongcount = 0

    
        
    for filename in os.listdir('examples/'):
        
        
        
        if filename.find(".gif")== -1:
            continue 
        try:
            im = Image.open("examples/%s"%(filename))

            print ""
            print filename
            #im = im.convert("RGBA")
            im = im.convert("P")
            im2 = Image.new("P", im.size, 255)



            temp = {}
            for x in range(im.size[1]):
                for y in range(im.size[0]):
                    pix = im.getpixel((y, x))
                    temp[pix] = pix
                    #if pix == 3:      
                    if pix == 220 or pix == 227: # these are the numbers to get

                        im2.putpixel((y, x), 0)

            inletter = False
            foundletter = False
            start = 0
            end = 0
            letters = []
            for y in range(im2.size[0]): # slice across
                for x in range(im2.size[1]): # slice down
                    pix = im2.getpixel((y, x))
                    if pix != 255:
                        inletter = True

                if foundletter == False and inletter == True:
                    foundletter = True
                    start = y

                if foundletter == True and inletter == False:
                    foundletter = False
                    end = y
                    letters.append((start, end))
                inletter = False

            count = 0
            countid = 1    
            word_sug = None
            end = time.time()
            elapsed = end - start

            for letter in letters:
                m = hashlib.md5()
                print "----------------------------\n"
                im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
                guess = []
                for image in imageset:
                    for x, y in image.iteritems():
                        if len(y) != 0:
                            guess.append(( v.relation(y[0], buildvector(im3)), x))
                guess.sort(reverse=True)
                word_per = guess[0][0] * 100
                if str(word_per) == "100.0":
                    print "Image position   :", countid
                    print "Broken Percent   :", int(round(float(word_per))), "%", "[+]"
                else:
                    print "Image position   :", countid
                    print "Broken Percent   :", "%.4f" % word_per, "%"
                print "------------------"
                print "Word suggested   :", guess[0][1]

                if word_sug == None:
                    word_sug = str(guess[0][1])
                else:
                    word_sug = word_sug + str(guess[0][1])
                count += 1
                countid = countid + 1




            if word_sug == filename[:-4]:
                correctcount += 1
            else:
                wrongcount += 1

        
        except:
            break


    print "======================="
    correctcount = float(correctcount)
    wrongcount = float(wrongcount)
    print "Correct Guesses - ",correctcount
    print "Wrong Guesses - ",wrongcount
    print "Percentage Correct - ", correctcount/(correctcount+wrongcount)*100.00
    print "Percentage Wrong - ", wrongcount/(correctcount+wrongcount)*100.00



if __name__ == '__main__':
   crack()




import cv2
import numpy as np
from datetime import datetime
import os


###########################################################################
    
#prepare data for training
rps=['4','6','6','X','R','H','E','K','1','A','R','I','B','8','6',
            '5','6','1','7','W','K','B','7','7','7','P','K','A','A','7',
            '1','2','5','Q','P','F','8','4','6','0','X','W','4','1','2',
            '5','5','B','A','7','8','4','A','A','8','2','0','B','A','E',
            'B','Y','7','0','3','7','M','4','7','3','R','Q','H','9','0',
            '2','7','E','I','X','S','1','D','D','3','3','1','4','5','6',
            'R','G','6','5','R','A','0','1','J','P','7','5','A','1','H',
            'G','4','2','3']
rps=[52, 54, 54, 88, 82, 72, 69, 75, 49, 65, 82, 73, 66, 56, 54, 53, 54, 49,
    55, 87, 75, 66, 55, 55, 55, 80, 75, 65, 65, 55, 49, 50, 53, 81, 80, 70, 56,
    52, 54, 48, 88, 87, 52, 49, 50, 53, 53, 66, 65, 55, 56, 52, 65, 65, 56, 50,
    48, 66, 65, 69, 66, 89, 55, 48, 51, 55, 77, 52, 55, 51, 82, 81, 72, 57, 48,
    50, 55, 69, 73, 88, 83, 49, 68, 68, 51, 51, 49, 52, 53, 54, 82, 71, 54, 53,
    82, 65, 48, 49, 74, 80, 55, 53, 65, 49, 72, 71, 52, 50, 51]
responses = []
print len(rps)
def prepare_data():
    global responses
    fileNumber=len(rps)
    counter=0;
    samples=np.empty((0,100))
    for f in range(fileNumber):
        im = cv2.imread('./numbers/'+str(f)+'.jpg')
        im3 = im.copy()
        height, width, depth = im3.shape
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   
        roi = thresh[0:height, 0:width]
        roismall = cv2.resize(roi,(10,10))
        responses.append(rps[f])
        sample = roismall.reshape((1,100)) 
        samples = np.append(samples,sample,0)
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "training complete"

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
    
##########################################################
def train():
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((1,responses.size))
    
    model = cv2.KNearest()
    model.train(samples,responses)
    print ' training complete'
    return model


prepare_data()
#train()
#image_processing()



'''
        im = Image.open("test2.png")
        im2 = Image.new("P", im.size, 255)
        im = im.convert("P")
'''
