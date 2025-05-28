from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
import datetime
import mysql.connector
import sys
import pickle
import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')

@app.route("/RainFall")
def RainFall():
    return render_template('RainFall.html')

@app.route("/Pesticides")
def Pesticides():
    return render_template('Pesticides.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')


@app.route("/NewQuery1")
def NewQuery1():
    return render_template('NewQueryReg.html')


@app.route("/UploadDataset")
def UploadDataset():
    return render_template('ViewExcel.html')


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/UserHome")
def UserHome():
    user = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + user + "'")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)


@app.route("/UQueryandAns")
def UQueryandAns():
    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
    data = cur.fetchall()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
    data1 = cur.fetchall()

    return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1)


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
        if request.form['uname'] == 'admin' or request.form['password'] == 'admin':

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb ")
            data = cur.fetchall()
            return render_template('AdminHome.html', data=data)

        else:
            return render_template('index.html', error=error)


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            alert = 'Username or Password is wrong'
            render_template('goback.html', data=alert)



        else:
            print(data[0])
            session['uid'] = data[0]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name1 = request.form['name']
        gender1 = request.form['gender']
        Age = request.form['age']
        email = request.form['email']
        pnumber = request.form['phone']
        address = request.form['address']

        uname = request.form['uname']
        password = request.form['psw']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO regtb VALUES ('" + name1 + "','" + gender1 + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + uname + "','" + password + "')")
        conn.commit()
        conn.close()
        # return 'file register successfully'

    return render_template('UserLogin.html')


@app.route("/newquery", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':
        uname = session['uname']
        preg = request.form['Precipitation']
        temp_max = request.form['temp_max']
        temp_min = request.form['temp_min']
        wind = request.form['wind']
        #location = request.form['select']

        nit = float(preg)
        pho = float(temp_max)
        po = float(temp_min)
        te = float(wind)
        # age = int(age)

        filename = 'prediction-rfc-model.pkl'
        classifier = pickle.load(open(filename, 'rb'))

        data = np.array([[nit, pho, po, te]])
        my_prediction = classifier.predict(data)
        print(my_prediction)

        crop = ''
        fertilizer = ''

        if my_prediction == 0:
            pre = 'drizzle'
            rec = 'rice, millet (like bajra), finger millet (ragi), and other short-duration, water-tolerant crops'


        elif my_prediction == 1:
            pre = 'rain'
            rec = 'rice , maize (corns), millets, pulses (like green gram and black gram), certain types of gourds, leafy greens, and amaranth'

        elif my_prediction == 2:
            pre = 'sun'
            rec = 'corn, sorghum, sunflowers, pumpkins, tomatoes, peppers, melons, and okra'

        elif my_prediction == 3:
            pre = 'snow'
            rec = 'kale, cabbage, broccoli, carrots, turnips, spinach, lettuce, beets, and radishes'
        elif my_prediction == 4:
            pre = 'foggy'
            rec = 'Lettuce, spinach, broccoli, kale, cabbage, blueberries, raspberries, gooseberries, pineapple guava'


        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Querytb VALUES ('','" + uname + "','" + preg + "','" + temp_max + "','" + temp_min + "','" + wind + "',"
                                                                                         "'" + pre + "','" + rec + "','nil')")
        conn.commit()
        conn.close()
        # return 'file register successfully'
        uname = session['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
        data = cur.fetchall()

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
        data1 = cur.fetchall()

        return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1)


@app.route("/excelpost", methods=['GET', 'POST'])
def excelpost():
    if request.method == 'POST':

        file = request.files['fileupload']
        file_extension = file.filename.split('.')[1]
        print(file_extension)
        # file.save("static/upload/" + secure_filename(file.filename))

        import pandas as pd
        import matplotlib.pyplot as plt
        df = ''
        if file_extension == 'xlsx':
            df = pd.read_excel(file.read(), engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(file.read())
        elif file_extension == 'csv':
            df = pd.read_csv(file)

        print(df)

        import seaborn as sns
        sns.countplot(df['weather'], label="Count")
        plt.savefig('static/images/out.jpg')
        iimg = 'static/images/out.jpg'

        # plt.show()

        # df = pd.read_csv("./Heart/Heartnew.csv")

        # def clean_dataset(df):
        # assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        # df.dropna(inplace=True)
        # indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        # return df[indices_to_keep].astype(np.float64)

        # df = clean_dataset(df)

        # print("Preprocessing Completed")
        print(df)

        # import pandas as pd
        import matplotlib.pyplot as plt

        # read-in data
        # data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

        import seaborn as sns
        sns.countplot(df['weather'], label="Count")
        plt.show()
        weather_mapping = {'drizzle': 0, 'rain': 1, 'sun': 2, 'snow': 3, 'fog': 4}
        df['weather'] = df['weather'].map(weather_mapping)

        df_copy = df.copy(deep=True)
        df_copy[['precipitation', 'temp_max', 'temp_min', 'wind']] = df_copy[
            ['precipitation', 'temp_max', 'temp_min', 'wind']].replace(0, np.NaN)

        # Model Building
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=['weather', 'date'])
        y = df['weather']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        clreport = classification_report(y_test, y_pred)

        print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

        Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
        Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

        # Creating a pickle file for the classifier
        filename = 'prediction-rfc-model.pkl'
        pickle.dump(classifier, open(filename, 'wb'))

        print("Training process is complete Model File Saved!")

        df = df.head(200)

        # read_csv(..., skiprows=1000000, nrows=999999)

        return render_template('ViewExcel.html', data=df.to_html(), dataimg=iimg, tacc=Tacc, testacc=Testacc,
                               report=clreport)


@app.route("/AdminQinfo")
def AdminQinfo():
    # uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult='waiting'")
    data = cur.fetchall()

    return render_template('AdminQueryInfo.html', data=data)


@app.route("/answer")
def answer():
    Answer = ''
    Prescription = ''
    id = request.args.get('lid')

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM Querytb where  id='" + id + "'")
    data = cursor.fetchone()

    if data:
        UserName = data[1]
        nitrogen = data[2]
        phosphorus = data[3]
        potassium = data[4]
        temperature = data[5]
        humidity = data[6]
        ph = data[7]
        rainfall = data[8]



    else:
        return 'Incorrect username / password !'

    nit = float(nitrogen)
    pho = float(phosphorus)
    po = float(potassium)
    te = float(temperature)
    hu = float(humidity)
    phh = float(ph)
    ra = float(rainfall)
    # age = int(age)

    filename = 'crop-prediction-rfc-model.pkl'
    classifier = pickle.load(open(filename, 'rb'))

    data = np.array([[nit, pho, po, te, hu, phh, ra]])
    my_prediction = classifier.predict(data)
    print(my_prediction)

    crop = ''
    fertilizer = ''

    if my_prediction == 0:
        Answer = 'Predict'
        crop = 'rice'

        fertilizer = '4 kg of gypsum and 1 kg of DAP/cent can be applied at 10 days after sowing'

    elif my_prediction == 1:
        Answer = 'Predict'
        crop = 'maize'
        fertilizer = 'The standard fertilizer recommendation for maize consists of 150 kg ha−1 NPK 14–23–14 and 50 kg ha−1 urea'
    elif my_prediction == 2:
        Answer = 'Predict'
        crop = 'chickpea'

        fertilizer = 'The generally recommended doses for chickpea include 20–30 kg nitrogen (N) and 40–60 kg phosphorus (P) ha-1. If soils are low in potassium (K), an application of 17 to 25 kg K ha-1 is recommended'

    elif my_prediction == 3:
        Answer = 'Predict'
        crop = 'kidneybeans'
        fertilizer = 'It needs good amount of Nitrogen about 100 to 125 kg/ha'

    elif my_prediction == 4:
        Answer = 'Predict'
        crop = 'pigeonpeas'
        fertilizer = 'Apply 25 - 30 kg N, 40 - 50 k g P 2 O 5 , 30 kg K 2 O per ha area as Basal dose at the time of sowing.'

    elif my_prediction == 5:
        Answer = 'Predict'
        crop = 'mothbeans'
        fertilizer = 'The applications of 10 kg N+40 kg P2O5 per hectare have proved the effective starter dose'
    elif my_prediction == 6:
        Answer = 'Predict'
        crop = 'mungbean'
        fertilizer = 'Phosphorus and potassium fertilizers should be applied at 50-50 kg ha-1'
    elif my_prediction == 7:
        Answer = 'Predict'
        crop = 'blackgram'
        fertilizer = 'The recommended fertilizer dose for black gram is 20:40:40 kg NPK/ha.'
    elif my_prediction == 8:
        Answer = 'Predict'
        crop = 'lentil'
        fertilizer = 'The recommended dose of fertilizers is 20kg N, 40kg P, 20 kg K and 20kg S/ha.'
    elif my_prediction == 9:
        Answer = 'Predict'
        crop = 'pomegranate'
        fertilizer = 'The recommended fertiliser dose is 600–700 gm of N, 200–250 gm of P2O5 and 200–250 gm of K2O per tree per year'

    elif my_prediction == 10:
        Answer = 'Predict'
        crop = 'banana'
        fertilizer = 'Feed regularly using either 8-10-8 (NPK) chemical fertilizer or organic composted manure'

    elif my_prediction == 11:
        Answer = 'Predict'
        crop = 'mango'
        fertilizer = '50 gm zinc sulphate, 50 gm copper sulphate and 20 gm borax per tree/annum are recommended'

    elif my_prediction == 12:
        Answer = 'Predict'
        crop = 'grapes'
        fertilizer = 'Use 3 pounds (1.5 kg.) of potassium sulfate per vine for mild deficiencies or up to 6 pounds (3 kg.)'

    elif my_prediction == 13:
        Answer = 'Predict'
        crop = 'watermelon'
        fertilizer = 'Apply a fertilizer high in phosphorous, such as 10-10-10, at a rate of 4 pounds per 1,000 square feet (60 to 90 feet of row)'

    elif my_prediction == 14:
        Answer = 'Predict'
        crop = 'muskmelon'
        fertilizer = 'Apply FYM 20 t/ha, NPK 40:60:30 kg/ha as basal and N @ 40 kg/ha 30 days after sowing.'

    elif my_prediction == 15:
        Answer = 'Predict'
        crop = 'apple'
        fertilizer = 'Apple trees require nitrogen, phosphorus and potassium,Common granular 20-10-10 fertilizer is suitable for apples'

    elif my_prediction == 16:
        Answer = 'Predict'
        crop = 'orange'
        fertilizer = 'Orange farmers often provide 5,5 – 7,7 lbs (2,5-3,5 kg) P2O5 in every adult tree for 4-5 consecutive years'

    elif my_prediction == 17:
        Answer = 'Predict'
        crop = 'papaya'
        fertilizer = 'Generally 90 g of Urea, 250 g of Super phosphate and 140 g of Muriate of Potash per plant are recommended for each application'

    elif my_prediction == 18:
        Answer = 'Predict'
        crop = 'coconut'
        fertilizer = 'Organic Manure @50kg/palm or 30 kg green manure, 500 g N, 320 g P2O5 and 1200 g K2O/palm/year in two split doses during September and May'

    elif my_prediction == 19:
        Answer = 'Predict'
        crop = 'cotton'
        fertilizer = 'N-P-K 20-10-10 per hectare during sowing (through the sowing machine)'

    elif my_prediction == 20:
        Answer = 'Predict'
        crop = 'jute'
        fertilizer = 'Apply 10 kg of N at 20 - 25 days after first weeding and then again on 35 - 40 days after second weeding as top dressing'

    elif my_prediction == 21:
        Answer = 'Predict'
        crop = 'coffee'
        fertilizer = 'Coffee trees need a lot of potash, nitrogen, and a little phosphoric acid. Spread the fertilizer in a ring around each Coffee plant'


    else:
        Answer = 'Predict'
        crop = 'Crop info not Found!'

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    cursor = conn.cursor()
    cursor.execute(
        "update Querytb set DResult='" + Answer + "', CropInfo='" + crop + "',Fertilizer='" + fertilizer + "' where id='" + str(
            id) + "' ")
    conn.commit()
    conn.close()

    conn3 = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    cur3 = conn3.cursor()
    cur3.execute("SELECT * FROM regtb where 	UserName='" + str(UserName) + "'")
    data3 = cur3.fetchone()
    if data3:
        phnumber = data3[4]
        print(phnumber)
        sendmsg(phnumber, "Predict Crop Name : " + crop + " For More info Visit in Site")

    # return 'file register successfully'
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting '")
    data = cur.fetchall()
    return render_template('AdminAnswer.html', data=data)


@app.route("/AdminAinfo")
def AdminAinfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1weathercropdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting'")
    data = cur.fetchall()

    return render_template('AdminAnswer.html', data=data)


@app.route("/rainpr", methods=['GET', 'POST'])
def rainpr():
    if request.method == 'POST':
        select = request.form['select']
        rain = request.form['rain']

        out = ''

        import pandas as pd
        file_path = 'Crop/Crop_recommendation.csv'
        df = pd.read_csv(file_path)

        # Filter rows where the label is 'rice'
        rice_data = df[df['label'] == select]
        # Calculate the average rainfall for 'rice'

        average_rainfall = rice_data['rainfall'].mean()
        if select=='rice':
            if int(rain) > int(average_rainfall):

                out = """Ensure proper drainage to avoid water stagnation. Use flood-resistant rice varieties. Apply nitrogen-based fertilizers to recover nutrient loss.. """
            else:
                out = 'No Action Need'
        elif select=='maize':

            if int(rain) > int(average_rainfall):

                out = """To manage maize crops during excessive rainfall, focus on improving drainage, choosing appropriate varieties, timely planting, using mulch, and considering drainage systems like raised beds or furrow irrigation to prevent waterlogging and root damage, while also monitoring for potential diseases that may thrive in wet conditions. """
            else:
                out = 'No Action Need'
        elif select=='chickpea':

            if int(rain) > int(average_rainfall):

                out = """Avoid waterlogging by raising soil beds. Use fungicides to prevent root rot. Harvest early if damage is severe to save the crop. """
            else:
                out = 'No Action Need'

        elif select=='kidneybeans':

            if int(rain) > int(average_rainfall):

                out = """Ironically, waterlogged roots also tend to be poor at conducting water through the plant system, reducing overall nutrient and water uptake """
            else:
                out = 'No Action Need'
        elif select=='pigeonpeas':

            if int(rain) > int(average_rainfall):

                out = """To manage a pigeon pea crop experiencing excessive rainfall (overwatering), focus on planting varieties suited to waterlogged conditions, employing proper drainage techniques like raised beds or ridge planting, and using cultural practices like intercropping to reduce water stress; if necessary, consider supplemental irrigation management to regulate water supply during critical growth stages """
            else:
                out = 'No Action Need'

        elif select=='mothbeans':

            if int(rain) > int(average_rainfall):

                out = """To manage a moth beans   crop experiencing excessive rainfall (overwatering), focus on planting varieties suited to waterlogged conditions, employing proper drainage techniques like raised beds or ridge planting, and using cultural practices like intercropping to reduce water stress; if necessary, consider supplemental irrigation management to regulate water supply during critical growth stages """
            else:
                out = 'No Action Need'

        elif select=='mungbean':

            if int(rain) > int(average_rainfall):

                out = """To mungbean a moth beans   crop experiencing excessive rainfall (overwatering), focus on planting varieties suited to waterlogged conditions, employing proper drainage techniques like raised beds or ridge planting, and using cultural practices like intercropping to reduce water stress; if necessary, consider supplemental irrigation management to regulate water supply during critical growth stages """
            else:
                out = 'No Action Need'

        elif select == 'blackgram':

            if int(rain) > int(average_rainfall):

                out = """ Ensure field drainage and apply bio-fungicides. Use organic mulching to reduce root stress and protect plants."""
            else:
                out = 'No Action Need'

        elif select == 'lentil':

            if int(rain) > int(average_rainfall):

                out = """Use elevated beds and drainage channels. Apply fungicides to manage root diseases. Check for waterborne pests like aphids and treat promptly"""
            else:
                out = 'No Action Need'

        elif select == 'pomegranate':

            if int(rain) > int(average_rainfall):

                out = """Remove stagnant water around trees. Prune damaged branches to reduce fungal spread. Apply copper-based fungicides to prevent rot."""
            else:
                out = 'No Action Need'

        #out = int(average_rainfall)

        print(f"Average rainfall for rice: {average_rainfall:.2f}")

        return render_template('RainFall.html', out=out)





@app.route("/rainprr", methods=['GET', 'POST'])
def rainprr():
    if request.method == 'POST':
        select = request.form['select']
        out =''

        if select == 'Aphids':
            out = """Use insecticidal soap or neem oil to control aphids. Thoroughly spray affected leaves to smother the pests. 
                    Regularly inspect plants for early signs of aphid infestation to take timely action."""
        elif select == 'True bugs':
            out = """True bugs usually don't cause serious harm. Keep plants healthy with proper watering and care. 
            Remove weeds and debris around the plants to reduce their habitat."""
        elif select == 'Caterpillars':
            out = """Use Bacillus thuringiensis (Bt) spray to control caterpillars. Handpick visible ones to prevent further damage. 
            Monitor plants regularly to spot caterpillars early before they cause significant damage."""
        elif select == 'Whiteflies':
            out = """Spray neem oil or horticultural oil to manage whiteflies. Introduce predators like green lacewings for natural control. 
            Place yellow sticky traps near crops to catch and monitor whiteflies."""
        elif select == 'Spider Mites':
            out = """Spray a mixture of water and neem oil to kill spider mites. Increase humidity around plants to discourage their growth. 
            Prune infested leaves to prevent the spread of spider mites to healthy parts of the plant."""
        elif select == 'Thrips':
            out = """Apply spinosad-based spray to control thrips. Use blue sticky traps to monitor and reduce their population. 
            Avoid excessive nitrogen fertilization, as it can attract thrips."""
        elif select == 'Mealybugs':
            out = """Spray diluted isopropyl alcohol (70%) on mealybugs. Prune heavily infested parts to prevent spread. 
            Wash plants gently with water to remove mealybugs and eggs."""
        elif select == 'Cutworms':
            out = """Sprinkle diatomaceous earth around plant bases to block cutworms. Protect seedlings with collars. 
            Till the soil before planting to expose and kill cutworm larvae."""
        elif select == 'Beetles':
            out = """Use pyrethrin-based sprays to kill beetles. Handpick beetles in the morning when they're less active. 
            Rotate crops regularly to reduce the chance of beetle infestations."""
        elif select == 'Leafhoppers':
            out = """Spray neem oil or insecticidal soap to control leafhoppers. Remove plant debris to reduce hiding spots. 
            Plant resistant crop varieties to minimize damage caused by leafhoppers."""
        elif select == 'Corn Borers':
            out = """Use insecticides like pyrethroids to manage corn borers. Destroy crop residues after harvest to reduce larvae. 
            Crop rotation can help break the pest’s life cycle."""
        elif select == 'Flea Beetles':
            out = """Apply neem oil or pyrethrin-based sprays to control flea beetles. Use row covers to protect young plants. 
            Mulching around plants can deter flea beetle activity."""
        elif select == 'Weevils':
            out = """Use neem-based pesticides or diatomaceous earth to control weevils. Remove and destroy infested grains or plants. 
            Store grains in airtight containers to prevent infestation."""
        elif select == 'Stink Bugs':
            out = """Use insecticidal soap sprays or neem oil to manage stink bugs. Handpick bugs when visible on crops. 
            Clear weeds and debris where stink bugs might hide."""
        elif select == 'Root Maggots':
            out = """Apply insecticides like spinosad to soil to kill root maggots. Remove plant debris and use crop rotation to 
            disrupt their life cycle. Ensure good drainage to prevent maggot development."""
        elif select == 'Slugs':
            out = """Use iron phosphate-based baits to control slugs. Place barriers like copper tape around plants to deter them. 
            Handpick slugs at night when they are most active."""
        elif select == 'Cabbage Loopers':
            out = """Spray Bacillus thuringiensis (Bt) or spinosad to control cabbage loopers. Remove damaged leaves to limit their spread. 
            Use floating row covers to keep loopers off plants."""
        elif select == 'Sawflies':
            out = """Apply insecticidal soap or neem oil to manage sawfly larvae. Handpick larvae and prune affected branches. 
            Attract birds to naturally control sawfly populations."""
        elif select == 'Japanese Beetles':
            out = """Use neem oil or pyrethrin sprays to kill Japanese beetles. Handpick and drop beetles into soapy water. 
            Use pheromone traps to monitor and reduce their numbers."""
        elif select == 'Fire Ants':
            out = """Use bait containing hydramethylnon to control fire ants. Pour boiling water into their mounds for an organic solution. 
            Keep the area clean to prevent new colonies from forming."""
        elif select == 'Cutworms':
            out = """Sprinkle diatomaceous earth around plant bases to block cutworms. Protect seedlings with collars. 
            Till the soil before planting to expose and kill cutworm larvae."""
        elif select == 'Ants':
            out = """Apply boric acid baits to control ants effectively. Seal any entry points into storage areas or fields. 
            Keep crops free of sugary residues to reduce attraction."""
        elif select == 'Hornworms':
            out = """Handpick hornworms from plants and destroy them. Use Bacillus thuringiensis (Bt) to control larvae effectively. 
            Attract natural predators like wasps to help manage their population."""
        elif select == 'Leaf Miners':
            out = """Use neem oil or spinosad spray to kill leaf miners. Remove and destroy infested leaves to prevent spread. 
            Apply sticky traps to monitor and reduce adult populations."""
        elif select == 'Earworms':
            out = """Apply Bacillus thuringiensis (Bt) or pyrethroid sprays to control earworms. Remove crop residues after harvest. 
            Plant resistant crop varieties for long-term management."""
        elif select == 'Crickets':
            out = """Use carbaryl-based insecticides to control crickets. Remove weeds and grass around the crops to reduce hiding spots. 
            Install barriers like sticky traps to capture crickets."""
        elif select == 'Silverfish':
            out = """Apply diatomaceous earth to areas where silverfish are present. Keep storage areas dry and clean to deter them. 
            Seal cracks and crevices to block entry points."""
        elif select == 'Ticks':
            out = """Use permethrin-based sprays to kill ticks effectively. Keep grass trimmed and weeds removed around crops. 
            Introduce natural predators like chickens to control ticks naturally."""
        elif select == 'Grasshoppers':
            out = """Use nosema locustae-based bait to reduce grasshoppers. Set up fine netting to block their access to crops. 
            Encourage natural predators like birds to control their population."""
        elif select == 'Mosquitoes':
            out = """Use insect growth regulators (IGRs) like methoprene to control mosquito larvae. Remove standing water near fields 
            to prevent breeding. Plant citronella or marigold as natural mosquito repellents."""
        elif select == 'Midges':
            out = """Apply Bacillus thuringiensis (Bt) to water bodies to control midge larvae. Remove organic matter where they may breed. 
            Use light traps at night to reduce adult populations."""
        elif select == 'Locusts':
            out = """Use pesticides like fenitrothion to control locust swarms. Remove weeds and debris to reduce breeding sites. 
            Monitor early warning systems to prepare for potential infestations."""
        elif select == 'Earwigs':
            out = """Use diatomaceous earth around crops to deter earwigs. Place damp rolled-up newspapers near plants as traps, then dispose 
            of the earwigs. Maintain cleanliness to reduce their hiding spots."""
        elif select == 'Millipedes':
            out = """Spray insecticides like bifenthrin to control millipedes. Remove mulch or decaying organic matter near plants. 
            Keep soil well-drained to discourage their presence."""

        else:
            out = ''

        return render_template('Pesticides.html', out=out)
def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
