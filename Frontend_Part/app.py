import pandas as pd
import numpy as np
from flask import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from fpdf import FPDF
from imblearn.over_sampling import SMOTE
import mysql.connector

db = mysql.connector.connect(
    user="root", password="", port='3306', database='hate_speech')
cur = db.cursor()

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Hepatitis Test', 0, 2, 'C')

app = Flask(__name__)
app.secret_key = "######################"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/userhome')
def userhome():
    return render_template('userhome.html')


@app.route('/registration', methods=["POST", "GET"])
def registration():
    if request.method == 'POST':
        username = request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']

        if userpassword == conpassword:
            sql = "select * from user where Email='%s' and Password='%s'" % (
                useremail, userpassword)
            cur.execute(sql)
            data = cur.fetchall()
            db.commit()
            print(data)

            if data == []:
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val = (username, useremail, userpassword, Age, contact)
                cur.execute(sql, val)
                db.commit()
                flash("Registered successfully", "success")
                return render_template("login.html")
            else:
                flash("Details are invalid", "warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        useremail = request.form['useremail']
        session['useremail'] = useremail
        userpassword = request.form['userpassword']
        sql = "select * from user where Email='%s' and Password='%s'" % (
            useremail, userpassword)
        cur.execute(sql)
        data = cur.fetchall()
        db.commit()
        if data == []:
            msg = "user Credentials Are not valid"
            return render_template("login.html", name=msg)
        else:
            return render_template("userhome.html", myname=data[0][1])
    return render_template('login.html')


@app.route('/view')
def view():
    global df
    df = pd.read_excel("Dataset.xlsx")
    # selecting the first 100 records from the data
    dataset = df.head(100)
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer, df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        # read the data to pandas dataframe
        df = pd.read_excel('Dataset.xlsx')
        df.head()

        # assigning the input and output variable
        x = df.drop(['Category'], axis=1)
        y = df['Category']

        # Applying SMOTE
        smote = SMOTE(random_state=1)
        x , y = smote.fit_resample(x,y)

        # Applying PCA
        selected_columns = ['A/G RATIO', 'ALBUMIN SERUM', 'GLOBULIN(SERUM)', 'Total Bilirubin', 'Direct Bilirubin', 'Indirect Bilirubin', 'SGOT-ASG', 'TOTAL PROTEIN SERUM', 'ALKALINE PHOSPHATE']

        # Select the columns from the dataframe
        x = df[selected_columns]
        
        # Standardize the data (important for PCA)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # Perform PCA with 7 components
        n_components = 7
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_scaled)
        
        # Create a new dataframe with the original column names and the PCA components
        pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=x_pca, columns=pca_columns)

        # Again Split pca data into x and y
        x = pca_df
        y = df['Category']

        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=size, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

        print(x_train, x_test)
        print(y_train)
        print(y_test)
        
        y_train_indices = y_train.index
        
        # Get the first 20 indices from y_train_indices
        first_20_indices = y_train_indices[:20] 
        test_sample = df.loc[first_20_indices]
        print(test_sample)
        return render_template('preprocess.html', msg=f'Data Preprocessed and It Splits Successfully. Number of y_test: {len(y_test)}.', tabledata=test_sample.to_html(classes='data', header="true"))
    
    return render_template('preprocess.html')


@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model, ac_lr1
        df = pd.read_excel('Dataset.xlsx')
        df.head()

        # assigning the input and output variable
        x = df.drop(['Category'], axis=1)
        y = df['Category']

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y, test_size=0.2, random_state=42)
        print('******************************************************************************************')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('*************************************************************************************')
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            ac_lr = accuracy_score(y_test, y_pred)
            ac_lr = ac_lr * 100
            msg = 'The accuracy obtained by Logistic Regression is  ' + \
                str(ac_lr) + str('%')
            return render_template('model.html', msg=msg)
        
        elif s == 2:
            classifier = DecisionTreeClassifier()
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + \
                str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        
        elif s == 3:
            classifier = RandomForestClassifier()
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + \
                str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)

        elif s == 4:
            classifier = LinearSVC()
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Support Vector Machine Classifier is ' + \
                str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')



@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    global name, f1,f2,f3,f4,f5,f6,f7,f8,f9,result,msg
    if request.method == "POST":
        name = request.form['f10']
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        
        print(f1)
        li = [[f1,f2,f3,f4,f5,f6,f7,f8,f9]]
        print(li)
        df = pd.read_excel('Dataset.xlsx')
        df.head()

        # assigning the input and output variable
        x = df.drop(['Category'], axis=1)
        y = df['Category']

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        result = model.predict(li)
        result = result[0]

        if result == 0:
            msg = '''The person doesn't have Hepatitis Disease'''
        else:
            msg = 'The person has been affected with Hepatitis Disease'
        return render_template('prediction.html', msg = msg)
    
    return render_template('prediction.html')


@app.route('/generate_pdf', methods=['POST','GET'])
def generate_pdf():

    # Generate PDF
    pdf_file = 'Lab Report.pdf'
    pdf = PDF()
    pdf.add_page()

    # Set font and size for content
    pdf.set_font('Arial', '', 12)

    # Add heading
    pdf.header()
    print('#################')
    print(msg)

    # Add labels and form values
    pdf.cell(0, 10, 'Name: {}'.format(name), 0, 1)
    pdf.cell(0, 10, 'A/G Ratio: {}'.format(f1), 0, 1)
    pdf.cell(0, 10, 'ALBUMIN SERUM: {}'.format(f2), 0, 1)
    pdf.cell(0, 10, 'GLOBULIN(SERUM): {}'.format(f3), 0, 1)
    pdf.cell(0, 10, 'Total Bilirubin: {}'.format(f4), 0, 1)
    pdf.cell(0, 10, 'Direct Bilirubin: {}'.format(f5), 0, 1)
    pdf.cell(0, 10, 'Indirect Bilirubin: {}'.format(f6), 0, 1)
    pdf.cell(0, 10, 'SGOT-ASG: {}'.format(f7), 0, 1)
    pdf.cell(0, 10, 'TOTAL PROTEIN SERUM: {}'.format(f8), 0, 1)
    pdf.cell(0, 10, 'ALKALINE PHOSPHATE: {}'.format(f9), 0, 1)
    pdf.cell(0, 10, 'Result: {}'.format(msg), 0, 1)
    
    pdf.output(pdf_file)

    #  Open PDF in browser
    
    import os
    os.system('open ' + pdf_file)

    #  Open PDF in default PDF Viewer
    # Provide the PDF for download
    return send_file(pdf_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
