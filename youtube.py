from flask import Flask,render_template,request,redirect,jsonify
import re
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
import json
import pickle

with open("config.json", "r") as c:
    params = json.load(c)["params"]

vector = pickle.load(open('vector.pkl', 'rb'))  # text to vector conversion
model = pickle.load(open('model.pkl', 'rb'))    # sentiment classification model
google_key = params['google_key']
genai.configure(api_key = google_key)
gpt_model = pickle.load(open("gptimp.pkl","rb")) # importent suggestion modle
labels = ['appreciation', 'normal', 'question', 'suggestion', 'trolling']

youtube = Flask(__name__)

if params["local_server"] == "True":
    print("server got....")
    youtube.config["SQLALCHEMY_DATABASE_URI"] = params["local_url"]
else:
    youtube.config["SQLALCHEMY_DATABASE_URI"] = params["prod_url"]

db = SQLAlchemy(youtube)


class comments(db.Model):
    id = db.Column(db.Integer, nullable=False,primary_key=True)
    comment = db.Column(db.String(1000))
    category = db.Column(db.String(20))


def text_cleaning(text):
    exclude = '!"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'
    pattern = re.compile('<.*?>').sub(r' ', text)  # remove html tag
    pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+').sub(r' ',
                                                                                                 pattern)  # remove links
    pattern = pattern.translate(str.maketrans("", "", exclude))  # remove symbols
    pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002500-\U00002BEF"  # chinese char
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         u"\U0001f926-\U0001f937"
                         # u"\U00010000-\U0010ffff"
                         u"\u2640-\u2642"
                         u"\u2600-\u2B55"
                         u"\u200d"
                         u"\u23cf"
                         u"\u23e9"
                         u"\u231a"
                         u"\ufe0f"  # dingbats
                         u"\u3030"
                         u"\u23F2"
                         "]+", flags=re.UNICODE).sub(r'', pattern)
    pattern = re.sub('[^a-zA-Z]', ' ', pattern)  # remove number
    pattern = " ".join(pattern.split()).lower()  # convert into lower and multipal
    return pattern

@youtube.route('/',methods = ['GET',"POST"])
def home():
    data = comments.query.all()
    return render_template('index.html', data = data,total = len(data))

@youtube.route('/add_comment', methods=['POST'])
def add_comment():
    comment = request.form['comment']
    sent = text_cleaning(comment)
    if len(sent) == 0:
        new_row = comments(comment=comment, category='olang')
        db.session.add(new_row)
        db.session.commit()
    else:
        pred = model.predict(vector.transform([sent]))
        category = labels[pred[0]]
        new_row = comments(comment=comment, category = category)
        db.session.add(new_row)
        db.session.commit()

    return redirect('/')

@youtube.route('/classify',methods=['GET','POST'])
def classify():
    appreciation = comments.query.filter_by(category = 'appreciation').all()
    question = comments.query.filter_by(category = 'question').all()
    suggestion = comments.query.filter_by(category = 'suggestion').all()
    trolling =comments.query.filter_by(category = 'trolling').all()
    olang = comments.query.filter_by(category = 'olang').all()
    normal = comments.query.filter_by(category='normal').all()
    alen = len(appreciation)
    qlen = len(question)
    slen = len(suggestion)
    tlen = len(trolling)
    olen = len(olang)
    nlen = len(normal)
    value = [alen,qlen,slen,tlen,nlen,olen]
    label = ['appreciation','question','suggestion','trolling','normal','olang']
    #
    # # impsugg = impsuggestion(suggestion)

    return  render_template('classify.html',appreciation=appreciation,normal=normal,question=question,suggestion=suggestion,trolling=trolling,olang=olang,
                            alen = alen,qlen = qlen,slen = slen,tlen = tlen,olen = olen,nlen=nlen,value = value,label = label )#impsugg = impsugg)


@youtube.route('/impsuggestion',methods = ['GET','POST'])
def impsuggestion():
        sugg = comments.query.filter_by(category='suggestion').all()
        sugg_list = [i.comment for i in sugg]
        prompt = f"""
                     return   important points,keysentence,repeated same meaning comment , by analysing whole given comments to you on which an youtuber can make content base on gives comments from viewer.
                     In your output, only return list that focus to give unique point and dont give any headlines like 'importent points" or useless content.
                      .do not give any headlines.

                                  {sugg_list}

                                  """
        response = gpt_model.generate_content(prompt)

        # Clean the data by stripping the backticks
        list_data = response.text.strip("")
        list_data = ",".join(list_data.split('\n'))
        # Load the cleaned data and convert to DataFrame
        return jsonify({'result': list_data})

@youtube.route('/impquestion',methods = ['GET','POST'])
def impquestion():
        que = comments.query.filter_by(category='question').all()
        que_list = [i.comment for i in que]
        prompt = f"""
                     return  important points,keysentence,repeated same meaning comment , by analysing whole given comments to you on which an youtuber can gives answer in next video.
                     In your output, only return list that focus to give only unique points and dont give any headlines like 'importent points" or useless content.

                                  {que_list}

                                  """
        response = gpt_model.generate_content(prompt)

        # Clean the data by stripping the backticks
        list_data = response.text.strip("")
        list_data = ",".join(list_data.split('\n'))
        # Load the cleaned data and convert to DataFrame
        return jsonify({'result': list_data})




if __name__ == "__main__":
    youtube.run(debug=True,port=5000,use_reloader=False)