# NIMH - Labeling data re-use statements with Active Learning

There are three parts to this project and it includes the following:

* Data retrieval 
  * There are two classes, DatabaseIngest (ingest.py) and DownloadFiles (download.py) that are helper classes that provide methods to interact with a SQLite3 database and methods to interact with a REST API.
  * The createDataset.py is the class that inserts three CSV files downloaded from https://federalreporter.nih.gov/FileDownload (Projects, Publications, and Link Tables) into the database and using the SQL query (SELECT DISTINCT lt.pmid, lt.project_number FROM linktable AS lt INNER JOIN project AS p ON lt.project_number = p.project_number), download the research papers from the result. 
  * If you want to run this yourself, there are some changes to the code you need to make as there are some hard coded values in here tailored to my development environment.
   1. 
* Data analysis and modeling
