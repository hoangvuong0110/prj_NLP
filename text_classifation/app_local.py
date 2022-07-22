from flask import Flask, render_template, request
from keras.models import load_model
from underthesea import word_tokenize
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import regex
import pickle

model = load_model('model_text.h5')


class_names = ['Vãng lai','Đồ ăn ngon','Đồ ăn tệ','Đồ ăn bình thường','Giá cao','Giá tốt','Vệ sinh tốt','Không gian không tốt','Nhân viên tốt','Gọi món lâu']
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html', data="", class_names=enumerate(class_names), percents=None)


max_len = 117

with open('C:/Users/Hoang Vuong/OneDrive/Desktop/HK8/text_classifation/word_id.pkl', 'rb') as f:
    word_id = pickle.load(f)

# hàm xử lí câu
def xuli(cau):
  cau = regex.sub('[.,;/!?#“”$%()^&*-@+=]',' ',cau)
  cau = cau.lower()
  cau = word_tokenize(cau,format='text')
  cau = cau.strip()
  return cau

def chuyenCauThanhSo(cau):
  arr = []
  for i in cau.split():
    arr.append(word_id[i])
  return arr

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, class_names

    
    text = request.form['text_input']

    for cau in text.split('.'):
      cau = xuli(cau)
      s = ''
      for i in cau.split():
        if i in list(word_id.keys()):
          s+=i+' '
      s = s.strip()
      s = chuyenCauThanhSo(s)

    if len(s)>0:
      s = pad_sequences([s],maxlen=max_len,padding='post')
      pre = model.predict(s)
      prediction = [round(float(pre[0][i])*100, 3) for i in range(len(class_names))]  
      final = class_names[np.argmax(pre)]

    return render_template('index.html', data=final, class_names=enumerate(class_names), percents=prediction)


if __name__ == "__main__":
    app.run(debug=True)
