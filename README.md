# Deployed IR Prediction [Model](https://nguyenvu06-pls-test-app-pls-test-deployment-kdws9o.streamlitapp.com/)
A Dashboard to allow user to import a trained picked PLS model and make a near real time prediction. Model training and pickling can be done [here](https://nguyenvu06-partialleastsquares-irmodel-pls-visualization-dpbic7.streamlitapp.com/).

The current version hard coded the directory where the CSV spectra will be read from. Future version will allow the user to pick the local directory.

The application will apply a Savgol filter to the IR signal based on the model and generate prediction concentration
