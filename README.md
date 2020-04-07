# NIMH - Labeling data re-use statements with Active Learning

There are three parts to this project and it includes the following:

* Data retrieval 
  * There are two classes, DatabaseIngest (ingest.py) and DownloadFiles (download.py) that are helper classes that provide methods to interact with a SQLite3 database and methods to interact with a REST API, respectively.
  * The createDataset.py is the class that inserts three CSV files downloaded from https://federalreporter.nih.gov/FileDownload (Projects, Publications, and Link Tables) into the database and using the SQL query (SELECT DISTINCT lt.pmid, lt.project_number FROM linktable AS lt INNER JOIN project AS p ON lt.project_number = p.project_number), download the research papers from the result. 
  * If you want to run this yourself, there are some changes to the code you need to make as there are some hard coded values in here tailored to my development environment, mainly where to find the downloaded CSV files to read and where to write the output. Search for "/Users/G/..." and replace it with your local directory. The variable names should be self explanatory to indicate what that variable is for. One thing to note is that I keep a status file (it is just a simple text file) to see what have been downloaded so if anything goes wrong or if I need to kill the program, I can start where I left off instead of having to download the files again.
  * Once the files have been downloaded, the analysis can begin. There are still a lot of preprocessing and filtering of the data. Since this project explores the concept of active learning, an initial preselected labeled batch is used for the initial training of the model (it can be found in data/gilbert_data_reuse.csv)
  
* Data analysis and modeling

* Mobile app (written in React Native)
