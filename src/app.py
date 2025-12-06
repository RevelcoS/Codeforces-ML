from flask import Flask, render_template, request

from modules.model import Model, prediction_setup

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    result = ""
    statement = ""

    if request.method == 'POST':
        statement = request.form.get('statement', '')
        prediction = Model.predict_single(statement)[0]
        prediction = int(prediction)
        result = f'{prediction}'

    return render_template('index.html', statement=statement, result=result)

if __name__ == '__main__':

    print('Loading resources...')
    prediction_setup()

    app.run(debug=False)    
