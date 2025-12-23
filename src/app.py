from flask import Flask, render_template, request

from model.test import test_setup, predict_rating

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    result = ""
    statement = ""

    if request.method == 'POST':
        statement = request.form.get('statement', '')
        _, rating = predict_rating(statement)
        result = str(int(rating))

    return render_template('index.html', statement=statement, result=result)

if __name__ == '__main__':

    print('Setting up context...')
    test_setup()

    app.run(debug=False)    
