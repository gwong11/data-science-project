# NIMH - Labeling data reuse statements with Active Learning

NOTE: The papers are not in the repository, as the files were too big. Below shows how you could retrieve the paper using the scripts I have written, to do further analysis.

There are three parts to this project and it includes the following:

* Data retrieval 
  * There are two classes, DatabaseIngest (ingest.py) and DownloadFiles (download.py) that are helper classes that provide methods to interact with a SQLite3 database and methods to interact with a REST API, respectively.
  * The createDataset.py is the main file to run to retrieve data. It takes it options which, to see them, once the program runs, type 'h' to see the options. This file uses two other files ingest.py and download.py. The ingest.py contains a class called DatabaseIngest that consists of functions to interface with a SQLite3 database. The download.py contains a class called DownloadFiles which consists of functions to interface with an API.
  * The createDataset.py is the file that inserts three CSV files downloaded from https://federalreporter.nih.gov/FileDownload (Projects, Publications, and Link Tables) into the database and using the SQL query (SELECT DISTINCT lt.pmid, lt.project_number FROM linktable AS lt INNER JOIN project AS p ON lt.project_number = p.project_number), download the research papers from the result. 
  * If you want to run this yourself, there are some changes to the code you need to make as there are some hard coded values in here tailored to my development environment, mainly where to find the downloaded CSV files to read and where to write the output. Search for "/Users/G/..." and replace it with your local directory. The variable names should be self explanatory to indicate what that variable is for. One thing to note is that I keep a status file (it is just a simple text file) to see what have been downloaded so if anything goes wrong or if I need to kill the program, I can start where I left off instead of having to download the files again.
  * Once the files have been downloaded, the analysis can begin. There are still a lot of preprocessing and filtering of the data. Since this project explores the concept of active learning, an initial preselected labeled batch is used for the initial training of the model (it can be found in data/gilbert_data_reuse.csv)
  
* Data analysis and modeling
  * Similar to data retrieval, there is one main file called analysis.py which does some preliminary data analysis and also the file to create the machine learning model. There are two extra files defined for this stage. Everything is contained in this one file and allows input from the user. When the script runs, type 'h' to see what the options are.
  * Before machine learning, some preliminary analysis is done such as looking for missing data, etc. The script also produces a nice wordcloud to see what kind of data reuse are we looking at for the data retrieved in the data retrieval stage.
  * The majority of the file is for machine learning and exploring different vectorizers to normalize the data. Currently, two vectorizers supported are CountVectorizer and TfidfVectorizer. The script allows the user to either try to run the data against a series of normal machine learning algorithms (supported are LogisticRegression, MultinomialNB, SVM, and RandomForest) or choose active learning (which it will ask for more inputs). Depending on the queries the user entered, the program will iterate through the active learning query function and ask you to label whatever comes out. Then after the model learns of the new data, it saves the model to a file.
  * A lot of stuff could be improve such as allowing the user to input an existing model and retrain that model with new input.
  * A series of metrics such as accuracy, recall, precision, and F1 score are explored to measure the model performance. Also, a series of ROC graphs are produced in the process to visually see the model performance. Finally, a boxplot is used to compare the performance between the models.
  * The most important thing to come out of the active learning is that as the number of queries increases, the performance of the model increases. It also supports CountVectorizer and TfidfVectorizer and a variety of query strategies. Run the analysis.py script and type 'h' to learn more of the different parameters it supports.
 
* Mobile app (written in React Native)
  * NIMHDataLabeler is the React Native project where the code is located. To run, cd into the directory and do a ```npm start```. If you have expo installed on your phone, you can basically scan the QR code that appears after the code compiles and the app will automatically load on your phone. It supports auto-loading so any changes you make in the code, it will automatically load on your phone, as long as you're connected to the same network. If you don't have the expo app installed, you can run it in an emulator. Inside the Home.js, modify the fetch URL to point to the backend URL as it's currently configured to point to my local machine.
  * The backend is written in Python using Flask. The server code is in server.py. To run it locally on your machine, ```python server.py```. Also provided is a Dockerfile where you can build a docker image and deploy the backend in a virtual environment like AWS EC2. The image contains the necessary packages to run the server code. You can do ```docker build -t <server-name> .``` to build to image (make sure you're in the same directory as the Dockerfile if you're using '.' at the end or you can specify the Dockerfile) then use ```docker run``` to create the container. You can proxy the backend using wsgi and nginx but I won't be doing that here since it's used mainly within a team so running it under development is fine. For production, it's best to use a production server.
