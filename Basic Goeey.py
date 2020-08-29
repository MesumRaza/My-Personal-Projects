'''
A simple Gooey example. One required field, one optional.
'''

from argparse import ArgumentParser
from gooey import Gooey
from colored import stylize, attr, fg

import joblib

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

model = joblib.load("diabeteseModel.pkl")

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age):
	#print(payload)
	
	values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]
	
	headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
	
	input_variables = pd.DataFrame([values],
								columns=headers, 
								dtype=float,
								index=['input'])	
	
	# Get the model's prediction
	prediction = model.predict(input_variables)
	#print("Prediction: ", prediction)
	prediction_proba = model.predict_proba(input_variables)[0][1]
	#print("Probabilities: ", prediction_proba)

	ret = {"prediction":float(prediction),"prediction_proba":float(prediction_proba)}

	return ret


@Gooey(program_name="Diabetes Prediction Application",richtext_controls=True,terminal_font_size=16)
def main():
	parser = ArgumentParser()
	
	headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
	
	for i in headers:
		parser.add_argument(i,
							action='store',
							help="Description Goes Here")
						
	args = parser.parse_args()
	
	prediction_output=predict(args.Pregnancies, args.Glucose, args.BloodPressure, args.SkinThickness, args.Insulin,args.BMI, args.DiabetesPedigreeFunction, args.Age)
	
	prediction_value=prediction_output.get('prediction_proba')
	prediction_proba=str(round(float(prediction_output.get('prediction_proba')),1)*100)+'%'
	prediction_desc='Positive' if float(prediction_value) >0.4 else 'Negative'
	
	print("The Person is Diagnosed as Diabetes: "+stylize(prediction_desc, attr("bold")))
	print(stylize("The Prediction Confidence is: "+prediction_proba, fg("blue") + attr("bold")))
	


if __name__ == '__main__':
    main()
