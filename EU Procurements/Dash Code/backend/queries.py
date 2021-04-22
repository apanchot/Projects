import pandas as pd
from pymongo import MongoClient
from backend.DB import eu, db
import backend.precomputing as precomputing

#from DB import eu, db
#import precomputing

########################################################################################################################
countries = ['NO', 'HR', 'HU', 'CH', 'CZ', 'RO', 'LV', 'GR', 'UK', 'SI', 'LT',
             'ES', 'FR', 'IE', 'SE', 'NL', 'PT', 'PL', 'DK', 'MK', 'DE', 'IT',
             'BG', 'CY', 'AT', 'LU', 'BE', 'FI', 'EE', 'SK', 'MT', 'LI', 'IS']

def year_country_filter(bot_year, top_year, country_list):
    filter_ = {
        '$match': {
            '$and': [{"_id.year": {'$gte': bot_year}}, {"_id.year": {'$lte': top_year}}],
            "_id.country": { '$in': country_list }
                  }
              }

    return filter_

# the function below filters the procurement database
# by year (YEAR) and issuer countries (ISO_COUNTRY_CODE), matching the appropriate CPV_DIVISION.

def year_country_cpv_filter(bot_year, top_year, country_list,cpv=None):
    if cpv == None:
        filter_ = {
            '$match': {
                '$and': [{"YEAR": {'$gte': bot_year}}, {"YEAR": {'$lte': top_year}}],
                "ISO_COUNTRY_CODE": {'$in': country_list}}}
    else:
        filter_ = {
            '$match': {
                '$and': [{"YEAR": {'$gte': bot_year}}, {"YEAR": {'$lte': top_year}}],
                "ISO_COUNTRY_CODE": { '$in': country_list},
                "CPV_DIVISION":{'$eq':cpv}
                      }
                  }

    return filter_

# Apply the pipeline to retrieve 'cpv_division_description' from CPV collection,
# Save this value as a python dict 'description_dict' to be used later.

description_list = list(db.cpv.aggregate(precomputing.pipeline_cpv_div_desc))
description_dict = {}

for i in description_list:
     description_dict[i['_id']['cpv_division']] = i['_id']['cpv_division_description']

##################################################################################

#Queries for looking up the cpv description
lookup_cpv={"$lookup":{
           "from": "cpv_grouped",
           "localField": "_id",
           "foreignField": "_id.cpv_division",
           "as": "cpv_lookup"
        }}

unwind_cpv={"$unwind":"$cpv_lookup"}

###################################################################################

def ex0_cpv_example(bot_year=2008, top_year=2020):
    """
    Returns all contracts in given year 'YEAR' range and cap to 100000000 the 'VALUE_EURO'

    Expected Output (list of documents):
    [{'result': count_value(int)}]
    """

    def year_filter(bot_year, top_year):
        filter_ = {
            '$match': {
                '$and': [{'YEAR': {'$gte': bot_year}}, {'YEAR': {'$lte': top_year}}],
                'VALUE_EURO': {'$lt': 100000000}
            }}

        return filter_

    count = {
        '$count': 'result'
    }

    pipeline = [year_filter(bot_year, top_year), count]

    list_documents = list(eu.aggregate(pipeline))

    return list_documents


def ex1_cpv_box(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns five metrics, described below
    Result filterable by floor year, roof year and country_list

    Expected Output:
    (avg_cpv_euro_avg, avg_cpv_count, avg_cpv_offer_avg, avg_cpv_euro_avg_y_eu, avg_cpv_euro_avg_n_eu)

    Where:
    avg_cpv_euro_avg = average value of each CPV's division contracts average 'VALUE_EURO', (int)
    avg_cpv_count = average value of each CPV's division contract count, (int)
    avg_cpv_offer_avg = average value of each CPV's division contracts average NUMBER_OFFERS', (int)
    avg_cpv_euro_avg_y_eu = average value of each CPV's division contracts average VALUE_EURO' with 'B_EU_FUNDS', (int)
    avg_cpv_euro_avg_n_eu = average value of each CPV's division contracts average 'VALUE_EURO' with out 'B_EU_FUNDS' (int)
    """

    avg_cpv_euro_avg = None
    avg_cpv_count = None
    avg_cpv_offer_avg = None
    avg_cpv_euro_avg_y_eu = None
    avg_cpv_euro_avg_n_eu = None

    #### 1.1
    query0 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sum"}, "sumOfCounts": {"$sum": "$count"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$group": {"_id": {}, "avgValueEuro": {"$avg": "$avgValueEuro"}}}

    pipeline_1_1 = [year_country_filter(bot_year, top_year, country_list), query0,query1,query2]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_1_1)

    for document in myCursor:
        avg_cpv_euro_avg = document['avgValueEuro']
    #### 1.2
    query3 = {"$group": {"_id": "$_id.cpv_division",
                         "count": {"$sum": "$count"}}}

    query4 = {"$group": {"_id": {}, "avgCount": {"$avg": "$count"}}}
    pipeline_1_2 = [year_country_filter(bot_year, top_year, country_list), query3, query4]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_1_2)

    for document in myCursor:
        avg_cpv_count = document['avgCount']

    #### 1.3
    query5 = {"$group": {"_id": "$_id.cpv_division",
                         "count": {"$sum": "$count"}, "sumOfOffers": {"$sum": "$countOffers"}}}

    query6 = {"$project": {"_id": "$_id", "avgNoOffers": {"$divide": ["$sumOfOffers", "$count"]}}}

    query7 = {"$group": {"_id": {}, "avgNoOffers": {"$avg": "$avgNoOffers"}}}
    pipeline_1_3 = [year_country_filter(bot_year, top_year, country_list), query5, query6,query7]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_1_3)

    for document in myCursor:
        avg_cpv_offer_avg = document['avgNoOffers']

    #### 1.4
    query8 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sumValueBEuFundsYes"},
                         "sumOfCounts": {"$sum": "$countBEuFundsYes"}}}

    query9 = {"$project": {"_id": "$_id", "avgValueEuro_Funds": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query10 = {"$group": {"_id": {}, "avgValueEuro_Funds": {"$avg": "$avgValueEuro_Funds"}}}

    pipeline_1_4 = [year_country_filter(bot_year, top_year, country_list), query8, query9,query10]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_1_4)

    for document in myCursor:
        avg_cpv_euro_avg_y_eu = document['avgValueEuro_Funds']

    #### 1.5
    query11 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sumValueBEuFundsNo"},
                         "sumOfCounts": {"$sum": "$countBEuFundsNo"}}}

    query12 = {"$project": {"_id": "$_id", "avgValueEuro_Funds": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query13 = {"$group": {"_id": {}, "avgValueEuro_Funds": {"$avg": "$avgValueEuro_Funds"}}}

    pipeline_1_5 = [year_country_filter(bot_year, top_year, country_list), query11, query12,query13]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_1_5)

    for document in myCursor:
        avg_cpv_euro_avg_n_eu = document['avgValueEuro_Funds']

    return avg_cpv_euro_avg, avg_cpv_count, avg_cpv_offer_avg, avg_cpv_euro_avg_y_eu, avg_cpv_euro_avg_n_eu


def ex2_cpv_treemap(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the count of contracts for each CPV Division
    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{cpv: value_1, count: value_2}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = contract count of each CPV Division, (int)
    """

    query1 = {"$group": {"_id": "$_id.cpv_division",
                         "count": {"$sum": "$count"}}}

    pipeline_2 = [year_country_filter(bot_year, top_year, country_list), query1, lookup_cpv, unwind_cpv]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.cpv_euro_avg.aggregate(pipeline_2)

    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "count": document['count']})
    return list_documents


def ex3_cpv_bar_1(bot_year=2008, top_year=2020, country_list=countries):
    """
    Per CPV Division and get the average 'VALUE_EURO' return the highest 5 cpvs
    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{cpv: value_1, avg: value_2}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each CPV Division, (float)
    """
    query0 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sum"}, "sumOfCounts": {"$sum": "$count"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$sort": {"avgValueEuro": -1}}

    query3 = {"$limit": 5}

    pipeline_3 = [year_country_filter(bot_year, top_year, country_list),query0, query1, query2, query3, lookup_cpv, unwind_cpv]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_3)
    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "avg": document['avgValueEuro']})


    return list_documents


def ex4_cpv_bar_2(bot_year=2008, top_year=2020, country_list=countries):
    """
    Per CPV Division and get the average 'VALUE_EURO' return the lowest 5 cpvs
    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{cpv: value_1, avg: value_2}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each CPV Division, (float)
    """

    query0 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sum"}, "sumOfCounts": {"$sum": "$count"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$sort": {"avgValueEuro": 1}}

    query3 = {"$limit": 5}

    pipeline_4 = [year_country_filter(bot_year, top_year, country_list), query0, query1, query2, query3, lookup_cpv,
                  unwind_cpv]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_4)
    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "avg": document['avgValueEuro']})
    return list_documents


def ex5_cpv_bar_3(bot_year=2008, top_year=2020, country_list=countries):
    """
    Per CPV Division and get the average 'VALUE_EURO' return the highest 5 cpvs for contracts which recieved european funds ('B_EU_FUNDS')
    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{cpv: value_1, avg: value_2}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each CPV Division, (float)
    """

    query0 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sumValueBEuFundsYes"},
                         "sumOfCounts": {"$sum": "$countBEuFundsYes"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro_Funds": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$sort": {"avgValueEuro_Funds": -1}}

    query3 = {"$limit": 5}

    pipeline_5 = [year_country_filter(bot_year, top_year, country_list), query0, query1, query2, query3, lookup_cpv,
                  unwind_cpv]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_5)
    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "avg": document['avgValueEuro_Funds']})
    return list_documents


def ex6_cpv_bar_4(bot_year=2008, top_year=2020, country_list=countries):
    """
    Per CPV Division and get the average 'VALUE_EURO' return the highest 5 cpvs for contracts which did not recieve european funds ('B_EU_FUNDS')
    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{cpv: value_1, avg: value_2}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each CPV Division, (float)
    """
    query0 = {"$group": {"_id": "$_id.cpv_division", "sumOfSum": {"$sum": "$sumValueBEuFundsNo"},
                         "sumOfCounts": {"$sum": "$countBEuFundsNo"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro_Funds": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$sort": {"avgValueEuro_Funds": -1}}

    query3 = {"$limit": 5}

    pipeline_6 = [year_country_filter(bot_year, top_year, country_list), query0, query1, query2, query3, lookup_cpv,
                  unwind_cpv]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_6)
    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "avg": document['avgValueEuro_Funds']})

    return list_documents


def ex7_cpv_map(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the highest CPV Division on average 'VALUE_EURO' per country 'ISO_COUNTRY_CODE'

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{cpv: value_1, avg: value_2, country: value_3}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = highest CPV Division average 'VALUE_EURO' of country, (float)
    value_3 = country in ISO-A2 format (string) (located in iso_codes collection)
    """
    query0 = {"$group": {"_id": {"cpv": "$_id.cpv_division", "country": "$_id.country"}, "sumOfSum": {"$sum": "$sum"},
                         "sumOfCounts": {"$sum": "$count"}}}

    query1 = {"$project": {"_id": "$_id", "avgValueEuro": {"$divide": ["$sumOfSum", "$sumOfCounts"]}}}

    query2 = {"$sort": {"_id.country": 1, "avgValueEuro": -1}}

    query3 = {"$group": {"_id": "$_id.country", "maxAvgValueEuro": {"$first": "$avgValueEuro"},
                         "cpv_division": {"$first": "$_id.cpv"}}}

    lookup_cpv_7 = {"$lookup": {
        "from": "cpv_grouped",
        "localField": "cpv_division",
        "foreignField": "_id.cpv_division",
        "as": "cpv_lookup"
    }}
    lookupCountry = { "$lookup": {
                                 "from": "iso_codes",
                                 "localField": "_id",
                                 "foreignField": "alpha-2",
                                 "as": "country_iso"
                                 }
                    }

    pipeline_7 = [year_country_filter(bot_year, top_year, country_list), query0, query1, query2, query3, lookup_cpv_7,
                  unwind_cpv, lookupCountry]

    myCursor = db.cpv_euro_avg.aggregate(pipeline_7)
    list_documents = []
    for document in myCursor:
        if document['_id'] != "UK":
            list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                                   "avg": document['maxAvgValueEuro'],
                                   "country": document['country_iso'][0]['name']
                                   })
        else:
            list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                                   "avg": document['maxAvgValueEuro'],
                                   "country":document["_id"]})

    return list_documents


def ex8_cpv_hist(bot_year=2008, top_year=2020, country_list=countries, cpv='50'):
    """
    Produce an histogram where each bucket has the contract counts of a particular cpv
     in a given range of values (bucket) according to 'VALUE_EURO'

     Choose 10 buckets of any partition
    Buckets Example:
     0 to 100000
     100000 to 200000
     200000 to 300000
     300000 to 400000
     400000 to 500000
     500000 to 600000
     600000 to 700000
     700000 to 800000
     800000 to 900000
     900000 to 1000000


    So given a CPV Division code (two digit string) return a list of documents where each document as the bucket _id,
    and respective bucket count.

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{bucket: value_1, count: value_2}, ....]

    Where:
    value_1 = lower limit of respective bucket (if bucket position 0 of example then bucket:0 )
    value_2 = contract count for thar particular bucket, (int)
    """
    query1 = {
        "$bucket": {
            "groupBy": "$VALUE_EURO",  # // Field to group by
            "boundaries": [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000,
                           10000000],  # // Boundaries
            "default": ">10Mil",  # // Bucket id for documents which do not fall into a bucket
            "output": {  # // Output for each bucket
                "count": {"$sum": 1}}}}

    pipeline_8 = [year_country_cpv_filter(bot_year, top_year, country_list,cpv),query1]

    myCursor = db.cpv_bucket.aggregate(pipeline_8)
    list_documents = []
    for document in myCursor:
        list_documents.append({"bucket": document['_id'],
                               "count": document['count']})

    return list_documents


def ex9_cpv_bar_diff(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the average time and value difference for each CPV, return the highest 5 cpvs

    time difference = 'DT-DISPATCH' - 'DT-AWARD'
    value difference = 'AWARD_VALUE_EURO' - 'VALUE_EURO'

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{cpv: value_1, time_difference: value_2, value_difference: value_3}, ....]

    Where:
    value_1 = CPV Division description, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'DT-DISPACH' - 'DT-AWARD', (float)
    value_3 = average 'EURO_AWARD' - 'VALUE_EURO' (float)
    """

    query1 = {"$group": {"_id": "$CPV",
                         "avgValueDiff": {"$avg": "$value"}, "avgTimeDiff": {"$avg": "$time"}}}

    query2 = {"$sort": {"avgTimeDiff": -1}}

    query3 = {"$limit": 5}

    pipeline_9 = [year_country_cpv_filter(bot_year, top_year, country_list), query1, query2,query3, lookup_cpv,
                  unwind_cpv]

    myCursor = db.cpv_time_diff_2.aggregate(pipeline_9)
    list_documents = []
    for document in myCursor:
        list_documents.append({"cpv": document['cpv_lookup']['cpv_division_description'],
                               "time_difference": document['avgTimeDiff'],
                               "value_difference":document['avgValueDiff']})
    return list_documents



def ex10_country_box(bot_year=2008, top_year=2020, country_list=countries):
    """
    We want five numbers, described below
    Result filterable by floor year, roof year and country_list

    Expected Output:
    (avg_country_euro_avg, avg_country_count, avg_country_offer_avg, avg_country_euro_avg_y_eu, avg_country_euro_avg_n_eu)

    Where:
    avg_country_euro_avg = average value of each countries ('ISO_COUNTRY_CODE') contracts average 'VALUE_EURO', (int)
    avg_country_count = average value of each countries ('ISO_COUNTRY_CODE') contract count, (int)
    avg_country_offer_avg = average value of each countries ('ISO_COUNTRY_CODE') contracts average NUMBER_OFFERS', (int)
    avg_country_euro_avg_y_eu = average value of each countries ('ISO_COUNTRY_CODE') contracts average VALUE_EURO' with 'B_EU_FUNDS', (int)
    avg_country_euro_avg_n_eu = average value of each countries ('ISO_COUNTRY_CODE') contracts average 'VALUE_EURO' with out 'B_EU_FUNDS' (int)
    """

    avg_country_euro_avg = 0
    avg_country_count = 0
    avg_country_offer_avg = 0
    avg_country_euro_avg_y_eu = 0
    avg_country_euro_avg_n_eu = 0



    filter_match = year_country_filter(bot_year, top_year, country_list)

    temp_filtered_country = { "$out" : "temp_filtered_country" }
    num_countries=len(country_list)
    db.business_euro_sum.aggregate([filter_match, temp_filtered_country])

    pre = {"$project": {
        "sum_no": 1,
        "count": 1,
        "sum": 1,
        "sumValueBEuFundsNo": 1,
        "sumValueBEuFundsYes": 1,
        "country": "$_id.country",
        "year": "$_id.year",
        "address": "$_id.address",
        "business": "$_id.business"

    }
    }

    # 10.1

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "averageAmount": {"$sum": "$sum"}, "count": {"$sum": "$count"}}}

    pipeline_10_1 = [pre, countryAvg_1]


    myCursor = db.temp_filtered_country.aggregate(pipeline_10_1)

    for document in myCursor:
        # print(document)
        avg_country_euro_avg += document['averageAmount'] / document['count']
    avg_country_euro_avg/=num_countries
    
    # 10.2

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "count": {"$sum": "$count"}}}

    countryAvg_2 = {"$group": {"_id": {}, "count": {"$avg": "$count"}}}


    pipeline_10_2 = [pre, countryAvg_1, countryAvg_2]


    myCursor = db.temp_filtered_country.aggregate(pipeline_10_2)

    for document in myCursor:
        avg_country_count = document['count']

    # 10.3

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "count": {"$sum": "$sum_no"}}}

    countryAvg_2 = {"$group": {"_id": {}, "count": {"$avg": "$count"}}}


    pipeline_10_3 = [pre, countryAvg_1, countryAvg_2]

    myCursor = db.temp_filtered_country.aggregate(pipeline_10_3)

    for document in myCursor:
        avg_country_offer_avg = document['count']

    # 10.4

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "sum": {"$sum": "$sumValueBEuFundsYes"},
                               "count": {"$sum": "$count"}}}

    pipeline_10_4= [pre, countryAvg_1]


    myCursor = db.temp_filtered_country.aggregate(pipeline_10_4)

    for document in myCursor:
        avg_country_euro_avg_y_eu += document['sum'] / document['count']

    avg_country_euro_avg_y_eu/=num_countries
    
    # 10.5

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "sum": {"$sum": "$sumValueBEuFundsNo"}, "count": {"$sum": "$count"}}}

    pipeline_10_5 = [pre, countryAvg_1]


    myCursor = db.temp_filtered_country.aggregate(pipeline_10_5)

    for document in myCursor:
        avg_country_euro_avg_n_eu += document['sum'] / document['count']
    avg_country_euro_avg_n_eu/=num_countries


    return avg_country_euro_avg, avg_country_count, avg_country_offer_avg, avg_country_euro_avg_y_eu, avg_country_euro_avg_n_eu

def ex11_country_treemap(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the count of contracts per country ('ISO_COUNTRY_CODE')
    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{country: value_1, count: value_2}, ....]

    Where:
    value_1 = Country ('ISO_COUNTRY_CODE') name, (string) (located in iso_codes collection')
    value_2 = contract count of each country, (int)
    """
    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "count": {"$sum": "$count"}}}

    pre = {"$project": {
        "count": 1,
        "country": "$_id.country",
        "year": "$_id.year",
        "address": "$_id.address",
        "business": "$_id.business"

    }
    }


    pipeline_10_1 = [year_country_filter(bot_year, top_year, country_list), pre, countryAvg_1]

    myCursor = db.business_euro_sum.aggregate(pipeline_10_1)

    list_documents = []
    for document in myCursor:
        list_documents.append({"country": document['_id']['country'], "count": document['count']})

    return list_documents


def ex12_country_bar_1(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the average 'VALUE_EURO' for each country, return the highest 5 countries

    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{country: value_1, avg: value_2}, ....]

    Where:
    value_1 = Country ('ISO_COUNTRY_CODE') name, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each country ('ISO_COUNTRY_CODE') name, (float)
    """

    pre = {"$project": {

        "count": 1,
        "sum": 1,

        "country": "$_id.country",
        "year": "$_id.year",
        "address": "$_id.address",
        "business": "$_id.business"

    }
    }

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "sum": {"$sum": "$sum"}, "count": {"$sum": "$count"}}}
    countryAvg_2 = {"$project": {"country": 1, "avgEuro": {"$divide": ["$sum", "$count"]}}}
    query2 = {"$sort": {"avgEuro": -1}}

    query3 = {"$limit": 5}

    pipeline_10_1 = [year_country_filter(bot_year, top_year, country_list), pre, countryAvg_1, countryAvg_2, query2, query3]

    myCursor = db.business_euro_sum.aggregate(pipeline_10_1)

    list_documents = []
    for document in myCursor:
        list_documents.append({"country": document['_id']['country'], "avg": document['avgEuro']})

    return list_documents


def ex13_country_bar_2(bot_year=2008, top_year=2020, country_list=countries):
    """
    Group by country and get the average 'VALUE_EURO' for each group, return the lowest, average wise, 5 documents

    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{country: value_1, avg: value_2}, ....]

    Where:
    value_1 = Country ('ISO_COUNTRY_CODE') name, (string) (located in cpv collection as 'cpv_division_description')
    value_2 = average 'VALUE_EURO' of each country ('ISO_COUNTRY_CODE') name, (float)
    """

    pre = {"$project": {

        "count": 1,
        "sum": 1,

        "country": "$_id.country",
        "year": "$_id.year",
        "address": "$_id.address",
        "business": "$_id.business"

    }
    }

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "sum": {"$sum": "$sum"}, "count": {"$sum": "$count"}}}
    countryAvg_2 = {"$project": {"country": 1, "avgEuro": {"$divide": ["$sum", "$count"]}}}
    query2 = {"$sort": {"avgEuro": 1}}

    query3 = {"$limit": 5}

    pipeline_10_1 = [year_country_filter(bot_year, top_year, country_list), pre, countryAvg_1, countryAvg_2, query2, query3]

    myCursor = db.business_euro_sum.aggregate(pipeline_10_1)

    list_documents = []
    for document in myCursor:
        list_documents.append({"country": document['_id']['country'], "avg": document['avgEuro']})

    return list_documents


def ex14_country_map(bot_year=2008, top_year=2020, country_list=countries):
    """
    For each country get the sum of the respective contracts 'VALUE_EURO' with 'B_EU_FUNDS'

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{sum: value_1, country: value_2}, ....]

    Where:
    value_1 = sum 'VALUE_EURO' of country ('ISO_COUNTRY_CODE') name, (float)
    value_2 = country in ISO-A2 format (string) (located in iso_codes collection)
    """

    pre = {"$project": {
        "sumValueBEuFundsYes": 1,
        "country": "$_id.country",
        "year": "$_id.year",
        "address": "$_id.address",
        "business": "$_id.business"

    }
    }

    lookupCountry = { "$lookup": {
                                 "from": "iso_codes",
                                 "localField": "_id.country",
                                 "foreignField": "alpha-2",
                                 "as": "country_iso"
                                 }
                    }

    countryAvg_1 = {"$group": {"_id": {"country": "$country"}, "sum": {"$sum": "$sumValueBEuFundsYes"}}}

    pipeline_10_1 = [year_country_filter(bot_year, top_year, country_list), pre, countryAvg_1, lookupCountry]

    myCursor = db.business_euro_sum.aggregate(pipeline_10_1)

    list_documents = []
    
    for document in myCursor:
        if document["_id"]["country"]!="UK":
            list_documents.append({
                "sum": document['sum'],
                "country": document['country_iso'][0]['name'],
                })
        else:
            list_documents.append({
                "sum": document['sum'],
                "country": "United Kingdom of Great Britain and Northern Ireland",
                })



    return list_documents

def ex15_business_box(bot_year=2008, top_year=2020, country_list=countries):
    """
    We want five numbers, described below

    Result filterable by floor year, roof year and country_list

    Expected Output:
    (avg_business_euro_avg, avg_business_count, avg_business_offer_avg, avg_business_euro_avg_y_eu, avg_business_euro_avg_n_eu)

    Where:
    avg_business_euro_avg = average value of each company ('CAE_NAME')  contracts average 'VALUE_EURO', (int)
    avg_business_count = average value of each company ('CAE_NAME') contract count, (int)
    avg_business_offer_avg = average value of each company ('CAE_NAME') contracts average NUMBER_OFFERS', (int)
    avg_business_euro_avg_y_eu = average value of each company ('CAE_NAME') contracts average VALUE_EURO' with 'B_EU_FUNDS', (int)
    avg_business_euro_avg_n_eu = average value of each company ('CAE_NAME') contracts average 'VALUE_EURO' with out 'B_EU_FUNDS' (int)
    """

    avg_business_euro_avg = None
    avg_business_count = None
    avg_business_offer_avg = None
    avg_business_euro_avg_y_eu = None
    avg_business_euro_avg_n_eu = None

    filter_match = year_country_filter(bot_year, top_year, country_list)

    temp_filtered = { "$out" : "temp_filtered" }

    db.business_euro_sum.aggregate([filter_match, temp_filtered])

    # Question 15.1
    # the queries below calculates the avg of each company ('CAE_NAME') contracts avg 'VALUE_EURO', (int)

    businessAvg_0 = { "$group": { "_id": {"per_business_sum": "$_id.business"},
                                  "total_sum": {"$sum": "$sum"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_1 = { "$project": {"total_sum": 1,
                                   "total_count": 1,
                                   "per_business_avg": { "$divide": [ "$total_sum", "$total_count" ] },
                                  }
                    }

    businessAvg_2 = { "$group": { "_id": "avg_business_euro_avg", "averageAmount": {"$avg": "$per_business_avg"} } }

    # resulting pipeline
    pipeline_15_1 = [businessAvg_0, businessAvg_1, businessAvg_2]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.temp_filtered.aggregate(pipeline_15_1)

    for document in myCursor:
        avg_business_euro_avg = document['averageAmount']

    # Question 15.2
    # the query below calculates the average of each company's contract count

    businessAvg_3 = { "$group": { "_id": "avg_business_count", "averageCount": {"$avg": "$total_count"} } }

    # resulting pipeline
    pipeline_15_2 = [businessAvg_0, businessAvg_3]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.temp_filtered.aggregate(pipeline_15_2)

    for document in myCursor:
        avg_business_count = document['averageCount']

    # Question 15.3
    # the queries below calculate the avarage of each company's contracts average NUMBER_OFFERS

    businessAvg_4 = { "$group": { "_id": {"per_business_sum_no": "$_id.business"},
                                  "total_sum_no": {"$sum": "$sum_no"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_5 = { "$project": {"total_sum_no": 1,
                                   "total_count": 1,
                                   "per_business_avg_no": { "$divide": [ "$total_sum_no", "$total_count" ] },
                                  }
                    }

    businessAvg_6 = { "$group": { "_id": "avg_business_offer_avg", "averageNumberOffers": {"$avg": "$per_business_avg_no"} } }

    # resulting pipeline
    pipeline_15_3 = [businessAvg_4, businessAvg_5, businessAvg_6]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.temp_filtered.aggregate(pipeline_15_3)

    for document in myCursor:
        avg_business_offer_avg = document['averageNumberOffers']

    # Question 15.4
    # the query below calculates the avarage of each company's contracts average 'VALUE_EURO’ with ‘B_EU_FUNDS’

    businessAvg_7 = { "$group": { "_id": {"per_business_sumValueBEuFundsYes": "$_id.business"},
                                  "total_sumValueBEuFundsYes": {"$sum": "$sumValueBEuFundsYes"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_8 = { "$project": {"total_sumValueBEuFundsYes": 1,
                                   "total_count": 1,
                                   "per_business_avgValueBEuFundsYes": { "$divide": [ "$total_sumValueBEuFundsYes", "$total_count" ] },
                                  }
                    }

    businessAvg_9 = { "$group": { "_id": "avg_business_euro_avg_y_eu", "averageAmountYes": {"$avg": "$per_business_avgValueBEuFundsYes"} } }

    # resulting pipeline
    pipeline_15_4 = [businessAvg_7, businessAvg_8, businessAvg_9]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.temp_filtered.aggregate(pipeline_15_4)

    for document in myCursor:
        avg_business_euro_avg_y_eu = document['averageAmountYes']

    # Question 15.5
    # the query below calculates the avarage of each company's contracts average 'VALUE_EURO’ without ‘B_EU_FUNDS’

    businessAvg_10 = { "$group": { "_id": {"per_business_sumValueBEuFundsNo": "$_id.business"},
                                  "total_sumValueBEuFundsNo": {"$sum": "$sumValueBEuFundsNo"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_11 = { "$project": {"total_sumValueBEuFundsNo": 1,
                                   "total_count": 1,
                                   "per_business_avgValueBEuFundsNo": { "$divide": [ "$total_sumValueBEuFundsNo", "$total_count" ] },
                                  }
                    }

    businessAvg_12 = { "$group": { "_id": "avg_business_euro_avg_n_eu", "averageAmountNo": {"$avg": "$per_business_avgValueBEuFundsNo"} } }

    # resulting pipeline
    pipeline_15_5 = [businessAvg_10, businessAvg_11, businessAvg_12]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.temp_filtered.aggregate(pipeline_15_5)

    for document in myCursor:
        avg_business_euro_avg_n_eu = document['averageAmountNo']

    return avg_business_euro_avg, avg_business_count, avg_business_offer_avg, avg_business_euro_avg_y_eu, avg_business_euro_avg_n_eu

def ex16_business_bar_1(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the average 'VALUE_EURO' for company ('CAE_NAME') return the highest 5 companies
    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{company: value_1, avg: value_2}, ....]

    Where:
    value_1 = company ('CAE_NAME') name, (string)
    value_2 = average 'VALUE_EURO' of each company ('CAE_NAME'), (float)
    """
    # Question 16
    # the queries below sort (descending) company's contracts average 'VALUE_EURO', (int)

    businessAvg_0 = { "$group": { "_id": {"per_business_sum": "$_id.business"},
                                  "total_sum": {"$sum": "$sum"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_1 = { "$project": {"total_sum": 1,
                                   "total_count": 1,
                                   "per_business_avg": { "$divide": [ "$total_sum", "$total_count" ] },
                                  }
                    }

    sortedAvg = { "$sort" : { "per_business_avg" : -1} }
    
    limitSorted = { "$limit" : 5 }

    # resulting pipeline
    pipeline_16 = [year_country_filter(bot_year, top_year, country_list),
                   businessAvg_0, businessAvg_1, sortedAvg, limitSorted]

    #pipeline_16 = [ businessAvg_0, businessAvg_1, sortedAvg, limitSorted]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.business_euro_sum.aggregate(pipeline_16)

    list_documents = []
    for document in myCursor:
        list_documents.append({"company": document['_id']['per_business_sum'], "avg": document['per_business_avg']})

    return list_documents

def ex17_business_bar_2(bot_year=2008, top_year=2020, country_list=countries):
    """
    Returns the average 'VALUE_EURO' for company ('CAE_NAME') return the lowest 5 companies


    Result filterable by floor year, roof year and country_list

    Expected Output (list of 5 sorted documents):
    [{company: value_1, avg: value_2}, ....]

    Where:
    value_1 = company ('CAE_NAME') name, (string)
    value_2 = average 'VALUE_EURO' of each company ('CAE_NAME'), (float)
    """

    # Question 17
    # the queries below sort (ascending) company's contracts average 'VALUE_EURO', (int)

    businessAvg_0 = { "$group": { "_id": {"per_business_sum": "$_id.business"},
                                  "total_sum": {"$sum": "$sum"},
                                  "total_count": {"$sum": "$count"},
                                }
                    }
    
    businessAvg_1 = { "$project": {"total_sum": 1,
                                   "total_count": 1,
                                   "per_business_avg": { "$divide": [ "$total_sum", "$total_count" ] },
                                  }
                    }

    sortedAvg = { "$sort" : { "per_business_avg" : 1} }
    
    limitSorted = { "$limit" : 5 }

    # resulting pipeline
    pipeline_17 = [year_country_filter(bot_year, top_year, country_list),
                   businessAvg_0, businessAvg_1, sortedAvg, limitSorted]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.business_euro_sum.aggregate(pipeline_17)

    list_documents = []
    for document in myCursor:
        list_documents.append({"company": document['_id']['per_business_sum'], "avg": document['per_business_avg']})

    return list_documents

def ex18_business_treemap(bot_year=2008, top_year=2020, country_list=countries):
    """
    We want the count of contracts for each company 'CAE_NAME', for the highest 15
    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{company: value_1, count: value_2}, ....]

    Where:
    value_1 = company ('CAE_NAME'), (string)
    value_2 = contract count of each company ('CAE_NAME'), (int)
    """

    # Question 18
    # the query below sorts (ascending) each company's count of contracts

    sortedCount = { "$sort" : { "count" : -1} }
    limitSorted = { "$limit" : 15 }

    # resulting pipeline
    pipeline_18 = [year_country_filter(bot_year, top_year, country_list), sortedCount, limitSorted]
   # pipeline_18 = [ sortedCount, limitSorted]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.business_euro_sum.aggregate(pipeline_18)

    list_documents = []
    for document in myCursor:
        list_documents.append({"company": document['_id']['business'], "count": document['count']})

    return list_documents


def ex19_business_map(bot_year=2008, top_year=2020, country_list=countries):
    """
    For each country get the highest company ('CAE_NAME') in terms of 'VALUE_EURO' sum contract spending

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{company: value_1, sum: value_2, country: value_3, address: value_4}, ....]

    Where:
    value_1 = 'top' company of that particular country ('CAE_NAME'), (string)
    value_2 = sum 'VALUE_EURO' of country and company ('CAE_NAME'), (float)
    value_3 = country in ISO-A2 format (string) (located in iso_codes collection)
    value_4 = company ('CAE_NAME') address, single string merging 'CAE_ADDRESS' and 'CAE_TOWN' separated by ' ' (space)
    """

    # Question 19

    sortedSum = { "$sort": { "_id.country" : 1, "sum" : -1} }
    
    firstInCountry = { "$group": {
                                "_id": "$_id.country", 
                                "sumEuro": { "$first": "$sum" },
                                "businessName": { "$first": "$_id.business"},
                                "businessAddress": { "$first": "$_id.address"},
                                }
                    }
    
    lookupCountry = { "$lookup": {
                                 "from": "iso_codes",
                                 "localField": "_id",
                                 "foreignField": "alpha-2",
                                 "as": "country_iso"
                                 }
                    }

    # resulting pipeline
    pipeline_19 = [year_country_filter(bot_year, top_year, country_list), sortedSum, firstInCountry, lookupCountry]

    # cursor from the pipeline above
    # it assumes the existence of a pre-calculated collection (defined in insert_operation)
    myCursor = db.business_euro_sum.aggregate(pipeline_19, allowDiskUse = True)

    list_documents = []
    
    for document in myCursor:
        if document['_id'] != "UK":
            list_documents.append({"company": document['businessName'],
                               "sum": document['sumEuro'],
                               "country": document['country_iso'][0]['name'],
                               "address": document['businessAddress']
                              })
        else:
            list_documents.append({"company": document['businessName'],
                               "sum": document['sumEuro'],
                               "country": "United Kingdom of Great Britain and Northern Ireland",
                               "address": document['businessAddress']
                              })

    return list_documents

def ex20_business_connection(bot_year=2008, top_year=2020, country_list=countries):
    """
    We want the top 5 most frequent co-occurring companies ('CAE_NAME' and 'WIN_NAME')

    Result filterable by floor year, roof year and country_list

    Expected Output (list of documents):
    [{companies: value_1, count: value_2}, ....]

    Where:
    value_1 = company ('CAE_NAME') string merged with company ('WIN_NAME') seperated by the string ' with ', (string)
    value_2 = co-occurring number of contracts (int)
    """
      
    group = {"$group": {"_id": {
                            "name": "$_id.co_name",
                           },
                        "total":{"$sum":"$count"},
                   }
            }
            
    sort = { "$sort" : { "total" : -1} }

    limit_20 = { "$limit" : 5 }
    
    pipeline_20 = [year_country_filter(bot_year, top_year, country_list), group, sort, limit_20]

    myCursor = db.business_co_occurrences.aggregate(pipeline_20, allowDiskUse = True)

    list_documents = []

    for document in myCursor:
        list_documents.append({"companies": document['_id']['name'], "count": document['total']})

    return list_documents

def insert_operation(document):
    '''
        Insert operation.

        In case pre computed tables were generated for the queries they should be recomputed with the new data.
    '''
    inserted_ids = eu.insert_many(document).inserted_ids

    # Apply the pipeline to set 'CPV_DIVISION' in the EU collection. 

    eu.update_many({}, precomputing.pipeline_add_cpv_division)

    # apply pipeline for 1-7
    eu.aggregate(precomputing.pipeline_q1_to_q7, allowDiskUse=True)
    
    # apply pipline for 8
    eu.aggregate(precomputing.pipeline_q8, allowDiskUse=True)

    # apply pipeline for 9
    eu.aggregate(precomputing.pipeline_q9, allowDiskUse=True)

    # Apply the pipeline to precompute the average VALUE_EURO for each combination
    # of CAE_NAME, YEAR and ISO_COUNTRY_CODE
    # (collection BUSINESS_EURO_SUM)

    eu.aggregate(precomputing.pipeline_companies_10_to_19, allowDiskUse = True)

    # Apply the pipeline to precompute the count of contracts for each combination
    # of CAE_NAME and WIN_NAME, year and country. The _id contains 'co_name' which is an aggregation of
    # CAE_NAME and WIN_NAME, year and country of the contracts.
    # (collection BUSINESS_CO_OCCURRENCES)

    eu.aggregate(precomputing.pipeline_co_occurrences_20, allowDiskUse = True)

    return inserted_ids

query_list = [
    ex1_cpv_box, ex2_cpv_treemap, ex3_cpv_bar_1, ex4_cpv_bar_2,
    ex5_cpv_bar_3, ex6_cpv_bar_4, ex7_cpv_map, ex8_cpv_hist ,ex9_cpv_bar_diff,
    ex10_country_box, ex11_country_treemap, ex12_country_bar_1,
    ex13_country_bar_2, ex14_country_map, ex15_business_box,
    ex16_business_bar_1, ex17_business_bar_2, ex18_business_treemap,
    ex19_business_map, ex20_business_connection
]
