from flask import Flask, render_template, request

app = Flask(__name__)
import pickle
import pandas as pd
import datetime

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()

current_time = datetime.datetime.now()
a = str(int(current_time.day) - 2).zfill(2)
date = str(current_time.month).zfill(2) + "-" + a + "-2021"
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv".format(date)
df = pd.read_csv(url, error_bad_lines=False)
new =  df[['Province_State', 'Country_Region', 'Confirmed','Active']]

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodypain = int(myDict['bodypain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        nasalcongestion = int(myDict['nasalcongestion'])
        diarrhoea = int(myDict['diarrhoea'])
        contact = int(myDict['contact'])
        drycough = int(myDict['drycough'])
        state = str(myDict['state'])
        active = new.loc[new['Province_State'] == state]['Active']
        inputFeatures = [fever, bodypain, age, runnyNose, diffBreath,drycough,diarrhoea,nasalcongestion,contact]
        infProb = clf.predict([inputFeatures])
        inf="aravind"
        if(infProb==0):
            inf="None"
        if (infProb == 1):
            inf="Mild"

        if (infProb == 2):
            inf="Mild"
        if (infProb == 3):
            inf="Severe"

        return render_template('show.html', inf=inf,state = state,active = int(active))
    return render_template('index_2.html')


# return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)
