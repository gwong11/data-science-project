from download import DownloadFiles
import ingest 
import time
import os
import pandas as pd

'''
The following consists of the headers required to retrieve the papers that 
are funded by the NIH:

Index(['PROJECT_ID', 'PROJECT_TERMS', 'PROJECT_TITLE', 'DEPARTMENT', 'AGENCY',
       'IC_CENTER', 'PROJECT_NUMBER', 'PROJECT_START_DATE', 'PROJECT_END_DATE',
       'CONTACT_PI_PROJECT_LEADER', 'OTHER_PIS', 'CONGRESSIONAL_DISTRICT',
       'DUNS_NUMBER', 'ORGANIZATION_NAME', 'ORGANIZATION_CITY',
       'ORGANIZATION_STATE', 'ORGANIZATION_ZIP', 'ORGANIZATION_COUNTRY',
       'BUDGET_START_DATE', 'BUDGET_END_DATE', 'CFDA_CODE', 'FY',
       'FY_TOTAL_COST', 'FY_TOTAL_COST_SUB_PROJECTS'],
      dtype='object')
Index(['AFFILIATION', 'AUTHOR_LIST', 'COUNTRY', 'ISSN', 'JOURNAL_ISSUE',
       'JOURNAL_TITLE', 'JOURNAL_TITLE_ABBR', 'JOURNAL_VOLUME', 'LANG',
       'PAGE_NUMBER', 'PMC_ID', 'PMID', 'PUB_DATE', 'PUB_TITLE', 'PUB_YEAR'],
      dtype='object')
Index(['PMID', 'PROJECT_NUMBER'], dtype='object')
'''

def create(conn):

    conn.create_table('project', """
                                  project_id PRIMARY KEY,
                                  project_terms,
                                  project_title,
                                  department, agency,
                                  ic_center,
                                  project_number NOT NULL,
                                  project_start_date,
                                  project_end_date,
                                  contact_pi_project_leader,
                                  other_pis,
                                  congressional_district,
                                  duns_number,
                                  organization_name,
                                  organization_city,
                                  organization_state,
                                  organization_zip,
                                  organization_country,
                                  budget_start_date,
                                  budget_end_date,
                                  cfda_code,
                                  fy,
                                  fy_total_cost,
                                  fy_total_cost_sub_projects 
                                  """)
    conn.create_table('publication', """
                                      affiliation,
                                      author_list,
                                      country,
                                      issn,
                                      journal_issue,
                                      journal_title,
                                      journal_title_abbr,
                                      journal_volume,
                                      lang,
                                      page_number,
                                      pmc_id,
                                      pmid PRIMARY KEY NOT NULL,
                                      pub_date,
                                      pub_title,
                                      pub_year 
                                      """)
    conn.create_table('linktable', """
                                      pmid PRIMARY KEY NOT NULL,
                                      project_number NOT NULL,
                                      FOREIGN KEY (pmid) REFERENCES publications (pmid),
                                      FOREIGN KEY (project_number) REFERENCES projects (project_number)
                                      """)

def insert(conn, year): 
    #start = time.time()

    proj_dir = '/Users/G/git/repository/data-science-project/data/projects'
    pub_dir = '/Users/G/git/repository/data-science-project/data/publications'
    link_dir = '/Users/G/git/repository/data-science-project/data/link_tables'

    # Get NIH related papers from projects
    proj_csv_file = os.path.join(proj_dir, 'FedRePORTER_PRJ_C_FY' + str(year) + '.csv')
    pub_csv_file = os.path.join(pub_dir, 'RePORTER_PUB_C_' + str(year) + '.csv')
    link_csv_file = os.path.join(link_dir, 'FedRePORTER_PUBLNK_C_FY' + str(year) + '.csv')

    proj_data = pd.read_csv(proj_csv_file, dtype='unicode')
    pub_data = pd.read_csv(pub_csv_file, dtype='unicode', encoding='latin-1')
    link_data = pd.read_csv(link_csv_file, dtype='unicode')

    proj_data.rename(columns=lambda x: x.strip(), inplace=True)
    pub_data.rename(columns=lambda x: x.strip(), inplace=True)
    link_data.rename(columns=lambda x: x.strip(), inplace=True)

    #print(proj_data.shape)
    # Only care about projects where IC_CENTER = 'NIMH' and 'AGENCY'
    proj_data = proj_data[proj_data['AGENCY'] == 'NIH']
    proj_data = proj_data[proj_data['IC_CENTER'] == 'NIMH']

    # Remove first character of the number and anything after a dash or space
    proj_data['PROJECT_NUMBER'] = proj_data['PROJECT_NUMBER'].str[1:]
    proj_data['PROJECT_NUMBER'].replace(to_replace='[- ][()a-zA-Z0-9]*', value='', regex=True, inplace=True)

    # drop duplicates
    proj_data.drop_duplicates(subset='PROJECT_NUMBER', keep='last', inplace=True)
    pub_data.drop_duplicates(subset='PMID', keep='last', inplace=True)
    link_data.drop_duplicates(subset='PMID', keep='last', inplace=True)

    #print(proj_data.shape)
    #print(proj_data[['PROJECT_NUMBER', 'IC_CENTER', 'AGENCY']].head(20))
    #print(pub_data[['AFFILIATION', 'PMC_ID', 'PMID']].head())
    #print(link_data[['PMID', 'PROJECT_NUMBER']].head())
    #print(proj_data.columns)
    #print(pub_data.columns)
    #print(link_data.columns)

    # Create the database
    print("Creating the database: project")
    conn.insert_record('project', 'append', proj_data)
    print("Done")

    print("Creating the database: publication")
    conn.insert_record('publication', 'append', pub_data)
    print("Done")

    print("Creating the database: linktable")
    conn.insert_record('linktable', 'append', link_data)
    print("Done")

    conn.commit()
    
    #print('It took ', time.time()-start, 'seconds.')

def check_PMID(pmid, status_file, action, found=False):

    present = False
    content = None
    fd = open(status_file, action)
    if 'a' in action:
        if found is False:
           fd.write(pmid + " *\n")
        else:
           fd.write(pmid + "\n")
    elif 'r' in action:
        content = fd.read()
        #print("Content: \n" + content)
        if pmid in content:
            present = True

    fd.close()
    return present 

def download(conn, paper_index, number_of_pmids):

    status_file = '/Users/G/git/repository/data-science-project/data/NIMH_pmids_downloads.txt'
    base_url = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_'
    output_dir = '/Users/G/git/repository/data-science-project/data/papers'

    sql_statement = "SELECT DISTINCT lt.pmid, lt.project_number FROM linktable AS lt INNER JOIN project AS p ON lt.project_number = p.project_number"

    df = conn.query_record(sql_statement)
    count = 0
    
    downloadObj = DownloadFiles(base_url)
    for pmid in df['pmid'].values:
        if os.path.exists(status_file) == False:
            print(str(count) + " - Attemtping to Download: " + pmid)
            found = downloadObj.retrieve(pmid)
            print("Creating status file: " + status_file)
            check_PMID(pmid, status_file, 'a+', found)
        elif check_PMID(pmid, status_file, 'r') is True:
            print(str(count) + " - Skipping (already attempted): " + pmid)
        elif check_PMID(pmid, status_file, 'r') is False:
            print(str(count) + " - Attempting to Download: " + pmid)
            found = downloadObj.retrieve(pmid)
            check_PMID(pmid, status_file, 'a+', found)

        if count == number_of_pmids:
            json_data = downloadObj.getPapersDict()
            if json_data:
               print("Writing output file: " + os.path.join(output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json'))
               downloadObj.write(downloadObj.getPapersDict(), output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json')
               downloadObj.clearPapersDict()
               paper_index += 1
               count = 0
        elif count == (len(df['pmid']) - 1):
            json_data = downloadObj.getPapersDict()
            if json_data:
               print("Writing final output file: " + os.path.join(output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json'))
               downloadObj.write(downloadObj.getPapersDict(), output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json')
               downloadObj.clearPapersDict()
        else:
            count += 1

def reprocess(conn, status_file, paper_index, number_of_pmids):

    reprocess_status_file = '/Users/G/git/repository/data-science-project/data/NIMH_pmids_reprocess_downloads.txt'
    reprocess_pmids = []

    fd = open(status_file, 'r')
    
    for line in fd.readlines():
        if "*" in line:
            reprocess_pmids.append(line.split(" ")[0])

    fd.close()

    base_url = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_'
    output_dir = '/Users/G/git/repository/data-science-project/data/papers'
    count = 0

    reprocessDownloadObj = DownloadFiles(base_url)
    for pmid in reprocess_pmids:
        if os.path.exists(reprocess_status_file) == False:
            print(str(count) + " - Attemtping to Reprocess: " + pmid)
            found = reprocessDownloadObj.retrieve(pmid)
            print("Creating reprocess status file: " + reprocess_status_file)
            check_PMID(pmid, reprocess_status_file, 'a+', found)
        elif check_PMID(pmid, reprocess_status_file, 'r') is True:
            print(str(count) + " - Skipping (already reprocessed): " + pmid)
        elif check_PMID(pmid, reprocess_status_file, 'r') is False:
            print(str(count) + " - Attempting to Reprocess: " + pmid)
            found = reprocessDownloadObj.retrieve(pmid)
            check_PMID(pmid, reprocess_status_file, 'a+', found)

        if count == number_of_pmids:
            json_data = reprocessDownloadObj.getPapersDict()
            if json_data:
               print("Writing reprocessed output file: " + os.path.join(output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json'))
               reprocessDownloadObj.write(reprocessDownloadObj.getPapersDict(), output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json')
               reprocessDownloadObj.clearPapersDict()
               paper_index += 1
               count = 0
        elif count == (len(reprocess_pmids) - 1):
            json_data = reprocessDownloadObj.getPapersDict()
            if json_data:
               print("Writing final reprocessed output file: " + os.path.join(output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json'))
               reprocessDownloadObj.write(reprocessDownloadObj.getPapersDict(), output_dir, 'NIMH_research_papers_' + str(paper_index) + '.json')
               reprocessDownloadObj.clearPapersDict()
        else:
            count += 1

def options(argument):

    switcher = {
        '1':  'Create',
        '2':  'Append',
        '3':  'Query',
        '4':  'Delete',
        '5':  'Drop',
        '6':  'Download',
        '7':  'Reprocess',
        'q':  'Quit',
        'h':  'Help'
    }

    return switcher.get(argument, "Invalid argument") 

def help():

    print("""
          Options available:
          1:  Create
          2:  Append
          3:  Query
          4:  Delete
          5:  Drop
          6:  Download
          7:  Reprocess
          q:  Quit
          h:  Help
          """)

if __name__ == '__main__':

    database = '/Users/G/git/repository/data-science-project/data/nimhresearch.db'
    conn = ingest.DatabaseIngest(database)
    conn.create_connection()

    arg = input("Enter option or h for help: ")
    while(arg != 'q'):
        if options(arg) == 'Help':
            help()
        elif options(arg) == 'Create':
            create(conn)
        elif options(arg) == 'Append':
            year = input("Enter year: ")
            insert(conn, year)
        elif options(arg) == 'Query':
            count = input("Enter number of results to display: ")
            sql = input("Enter sql statement: ")
            conn.query_record(int(count), sql)
        elif options(arg) == 'Delete':
            conn.delete_record('project')
            conn.delete_record('publication')
            conn.delete_record('linktable')
        elif options(arg) == 'Drop':
            conn.drop_table('project')
            conn.drop_table('publication')
            conn.drop_table('linktable')
        elif options(arg) == 'Download':
            paper_index = input("Enter starting index for creating files: ")
            number_of_pmids = input("Enter number of pmids to attempt: ")
            download(conn, int(paper_index), int(number_of_pmids))
        elif options(arg) == 'Reprocess':
            status_file = input("Enter status file (reprocessing): ")
            while os.path.exists(status_file) == False:
                status_file = input("File does not exist. Enter status file (reprocessing): ")
            paper_index = input("Enter starting index for creating files (reprocessing): ")
            number_of_pmids = input("Enter number of pmids to attempt (reprocessing): ")
            reprocess(conn, status_file, int(paper_index), int(number_of_pmids))

        arg = input("Enter option or h for help: ")

    conn.close()
