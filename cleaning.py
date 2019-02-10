import pandas as pd
import calendar

raw_data = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

# copy faw data
data = raw_data.copy()

#### 1. CHOOSING AN IDENTIFIER

data.shape # 31298 rows
data.columns # 26 columns

# Count the number of unique values per column
data.nunique()

'''
ID, UPA and Final Narrative have counts close to the number of columns.

ID                  31293
UPA                 31298
Final Narrative     31278
'''

identifier_cols = ['ID', 'UPA', 'Final Narrative']

data[identifier_cols].dtypes # int64, int64, object (string)
data[identifier_cols].head()

'''
Final Narrative is not elligible because it is unstructured data (text). Consider
dropping this as it can't be used for identification and also troublesome to structure.
'''

# drop Final Narrative
data = data.drop(columns=['Final Narrative'])

'''
Number of unique IDs do not match the number of rows in our dataset. Possibly due
to duplicates or errors. Consider dropping this in favour of UPA.

With UPA as the identifier column, we need to set it as a categorical variable
due to its numerical insignificance.
'''

# count rows with duplicate IDs
duplicate_ids = pd.concat(g for _, g in data.groupby("ID") if len(g) > 1)
len(duplicate_ids) # 10 rows

# drop ID
data = data.drop(columns=['ID'])

# change type to categorical
data["UPA"] = data["UPA"].astype('category')

#### 2. ANALYSIS OF LOCATION RELATED COLUMNS AS EVENT

location_cols = ['Address1', 'Address2', 'City', 'State', 'Zip', 'Latitude', 'Longitude']

data[location_cols].head()

data[location_cols].nunique()

'''
Address1 and Address2 provide little support in our association analysis due to
their respective number of unique values. Consider dropping these values in
favour of other location related columns with better support.

City and State can both be used in inferring location data, however, there are
less unique values in State and therfore might provide better support in our
association analysis.

With State as the chosen column to represent location events, we can now drop
City, Zip, Latitude and Longitude due to redundancy.
'''

# drop low support and redundant columns
data = data.drop(columns=['Address1', 'Address2', 'City', 'Zip', 'Latitude', 'Longitude'])

# change type of State to categorical
data["State"] = data["State"].astype('category')

data.columns

#### 3. ANALYSIS OF EVENTDATE AS EVENT

data['EventDate'].head()
data['EventDate'].describe() # date format is m/d/YYYY

# parse eventdate to datetime for ease of analysis
data['EventDate'] = pd.to_datetime(data['EventDate'], format='%m/%d/%Y')
data['EventDate'].describe() #  events from 1 Jan 2015 to 31 Jan 2018

'''
We can extract the Month, or Day of the Week out of the EventDate and use them
as events. Year would even be more unique, however, limited to only 3 values.
Would be pointless to use it since it's implication is past, unrepeatable events.
'''

# extract day of the week
data['Weekday'] = data['EventDate'].apply(lambda x: calendar.day_name[x.weekday()]).astype('category')

# extract month
data['Month'] = data['EventDate'].apply(lambda x: calendar.month_name[x.month]).astype('category')

# drop eventdate
data = data.drop(columns=['EventDate'])

#### 4. ANALYSIS OF EMPLOYER DATA AS EVENT

# See NAICS:  https://www.census.gov/eos/www/naics/faqs/faqs.html#q1
employer_cols = ['Employer', 'Primary NAICS']

data[employer_cols].describe()

'''
We can use the Primary NAICS column to represent the classification of the
employer's business. We can also use the Employer field but it would likely provide
less support due to the higher number of unique values.

To make use of the Primary NAICS, we need to map the codes into the textual
description.
'''

# drop employer column
data = data.drop(columns = ['Employer'])

# naics index dataset
naics = pd.read_csv('data/naics.csv')
naics.dtypes

'''
Before we can merge the two dataframes and perform the mapping, we need to make
sure there are no NA values in our data, and that both joining columns share
the same data type.
'''

data['Primary NAICS'] # dtype is object

data[data["Primary NAICS"].isnull()] # 2 rows have NA Primary NAICS

# drop those rows
data = data.dropna(subset=['Primary NAICS'])

is_numeric_naics = data['Primary NAICS'].str.isnumeric()
data[is_numeric_naics == False] # 1 row with NAICS as 48-49

# drop that column
data = data[is_numeric_naics]

# convert to int64
data['Primary NAICS'] = data["Primary NAICS"].astype('int64')

# merge dataframes
data = naics.merge(data, left_on='NAICS 1 Code', right_on='Primary NAICS')
data.head()

# remove join columns, rename description column
data = data.drop(columns = ['NAICS 1 Code', 'Primary NAICS'])
data = data.rename(columns={'NAICS 1 Description': 'NAICS'})
data['NAICS'] = data['NAICS'].astype('category')

#### 5. ANALYSIS OF COLUMNS WITH EXTREME NUMBER OF UNIQUE VALUES

extreme_unique_cols = ['Hospitalized', 'Amputation', 'Inspection']
data[extreme_unique_cols].nunique() # Hospitalized: 5, Amputation: 6, Inspection: 10212
data[extreme_unique_cols].describe()
data[extreme_unique_cols].dtypes # Amputation and Inspection are floats -> na?

# count na
len(data['Hospitalized']) - data['Hospitalized'].count() # 0
len(data['Amputation']) - data['Amputation'].count() # 1
len(data['Inspection']) - data['Inspection'].count() # 18910

# check hospitalized distribution
data.groupby('Hospitalized').size() # 1: 23267
data['Hospitalized'].hist()

# check amputation distribution
data.groupby('Amputation').size()[0] # 0: 21418
data['Amputation'].hist()

# check inspection distribution
data['Inspection'].describe()
data['Inspection'].hist()

'''
There's a very small number of unique values for Hospitalized and Amputation.
Reviewing the distribution, it might be better to categorize these columns as
'X' and 'not X'.

The column Inspection entails a significant portion of NA values. Similarly
we can do binary categorization here to increase the visibility of this event.

May be over the top, but we can investigate the single entry with NA Amputation
to decide if it should be 'Amputation' or 'No Amputation'.
'''

data[data['Amputation'].isnull()] # nature is avulsions, enucleations

'''
Avulsions: the action of pulling or tearing away.
Enucleations: surgically remove intact from its surrounding capsule.

x_x

Nature of the injury suggests that an amputation had occured. Sad.
'''

# fill amputation na as 1
data = data.fillna(value={'Amputation': 1})

# binary categorisation
data["Hospitalized"] = data["Hospitalized"].apply(lambda x: "Hospitalized" if x > 0 else "Not Hospitalized")
data["Amputation"] = data["Amputation"].apply(lambda x: "Amputation" if x > 0 else "No Amputation")
data.loc[~data["Inspection"].isnull(), "Inspection"] = "Has Inspection"
data["Inspection"] = data["Inspection"].fillna("No Inspection")

for col in extreme_unique_cols:
    data[col] = data[col].astype('category')

data[extreme_unique_cols].describe()

#### 5. ANALYSIS OF CODED COLUMNS RELATED TO THE INJURY

coded_cols = ['Nature', 'Part of Body', 'Event', 'Source', 'Secondary Source']
data[coded_cols].nunique()
data[coded_cols].dtypes # Secondary Source is float64 -> have na?

# check if any of the values in those columns are null
data[coded_cols].isnull().values.any() # True

# check if other cols except Secondary Source has na
data[coded_cols[:-1]].isnull().values.any() # False

# spikes indicate contentration on certain values
data.groupby('Nature').size().reset_index().plot(x='Nature', y=0)
data.groupby('Part of Body').size().reset_index().plot(x='Part of Body', y=0)
data.groupby('Event').size().reset_index().plot(x='Event', y=0)
data.groupby('Source').size().reset_index().plot(x='Source', y=0)
data.groupby('Secondary Source').size().reset_index().plot(x='Secondary Source', y=0)

'''
To yield better support for these fields, we can map the coded values into their
more generic classifications. This, in turn will render their descriptive counterpart
redundant, and can therefore be dropped.

Note that we need to handle Secondary Source differently as it has na values. Although
NA, these can still be used as events associated with Source.
'''

# can use an interation to retrieve mappings as the mapping files share the same columns
code_files = ['data/nature.csv', 'data/part.csv', 'data/events.csv', 'data/source.csv']
mappings = {} # store mappings here so can reuse later
for col, file in zip(coded_cols[:-1], code_files):
    code_data = pd.read_csv(file)

    # filter only top level hierarchy
    code_data = code_data[code_data['Hierarchy_level'] == 1]

    # only use CASE_CODE and CASE_CODE_TITLE
    code_data = code_data.reset_index()[['CASE_CODE', 'CASE_CODE_TITLE']]

    # convert to dictionary and add nonclassifiable type
    mapping = {k: code_data.loc[code_data['CASE_CODE'] == k, 'CASE_CODE_TITLE'].iloc[0] for k in code_data["CASE_CODE"]}
    mapping[9] = 'NONCLASSIFIABLE'
    mappings[col] = mapping

# transform columns to reflect more generic classification
for col in coded_cols[:-1]:
    data[col] = data[col].apply(lambda x: int(str(x)[:1])) # get leftmost digit
    data[col] = data[col].apply(lambda x: mappings[col][x]) # map

# handle Secondary Source differently because of NA values
with_secondary_source = data['Secondary Source'].notnull()
data.loc[with_secondary_source,'Secondary Source'] = data.loc[with_secondary_source,'Secondary Source'].apply(lambda x: mappings['Source'][int(str(x)[:1])])

# drop redundant code descriptors
data = data.drop(columns=['NatureTitle', 'Part of Body Title', 'EventTitle', 'SourceTitle', 'Secondary Source Title'])

# convert to categorical
for col in coded_cols:
    data[col] = data[col].astype('category')

data.head(1)

#### 6. RESHAPING FOR INPUT TO SAS

# verify
data.describe()
data.shape # 29183, 13

# calculate expected number of rows discounting those with na sec source
na_sec_source_count = len(data['Secondary Source']) - data['Secondary Source'].count() # 21373
data.shape[0] * data.shape[1] - na_sec_source_count # 358006 expected rows

# reshape data
rename_cols = { 'level_0':'UAP', 0:'Event' }
clean_data = data.stack().reset_index()[rename_cols.keys()].rename(columns=rename_cols)
clean_data.shape # 358006, 2

# export to csv
clean_data.to_csv('data/clean.csv')
