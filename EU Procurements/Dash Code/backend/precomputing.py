"""
This file contains the pipelines to precomputed tables
These pipelines will serve as input for the aggregations called
during the insertion of new documents
"""

"""
pipeline_add_cpv_division
(useful in questions 1 to 9)

Returns a collection named "cpv_grouped", grouping cpv div (in the cpv collection)
with only the first two digits (cpv division) and the cpv description

"""

switch={"$switch":{
    "branches":[
        {"case":{"$lt":["$CPV",1000000]},"then":"00"},
        {"case":{"$lt":["$CPV",10000000]},"then":{"$concat": ["0",{"$substr":["$CPV", 0, 1]}]}}
    ],
    "default": {"$substr":["$CPV", 0, 2]}
}}

add_cpv_division = {"$set":{
                            "CPV_DIVISION":switch
                           }
                   }

pipeline_add_cpv_division = [add_cpv_division]
"""
Create the CPV_Grouped collection, consisting in 2 fields: cpv division and cpv division description.
This collection allows for easier and faster lookup to retrieve the cpv division description field.
"""
query1={"$project":{"cpv_division":1,"cpv_division_description":1}}

query2={"$group": {"_id":{"cpv_division":"$cpv_division"},"cpv_division_description":{"$addToSet":"$cpv_division_description"}}}

query3={"$unwind":"$_id"}

query4={"$unwind":"$cpv_division_description"}

query5={"$out":"cpv_grouped"}

pipeline=[query1, query2,query3,query4,query5]

"""
pipeline_cpv_div_desc
(useful in questions 1 to 9)

Groups the CPV collection by cpv_division and cpv_division_description
This group will later be transformed into a python dict to be used when
answering to questions. The questions can be solved using cpv_division and
the final 'list_documents' can be updated with cpv_division_description.

"""
### WE ARE NOT USING THIS
pipeline_cpv_div_desc = [
    {"$group":
        {"_id":
            {"cpv_division": "$cpv_division",
             "cpv_division_description": "$cpv_division_description"
            },
        }
    }
]

"""
pipeline_1_to_7
Pipeline created to answer questions from 1 to 7.
Creates a new collection called cpv_euro_avg, which contains the fields to solve questions from 1 to 7.
"""
pipeline_q1_to_q7 = [
                # CAE_NAME must be string, the others must be not null
                {"$match": {"CPV_DIVISION": {"$type":2},
                            "VALUE_EURO": {"$ne":"null","$lte" : 100000000},
                            "YEAR": {"$ne":"null"},
                            "ISO_COUNTRY_CODE": {"$ne":"null"} }
                },
                # project only the necessary fields
                {"$project": {"CPV_DIVISION": 1,
                                "VALUE_EURO": 1,
                                "ISO_COUNTRY_CODE": 1,
                                "YEAR": 1,
                                "NUMBER_OFFERS": 1,
                                "bEuFundsYesCount":{
                                                # Set to VALUE_EURO if B_EU_FUNDS = Y
                                            "$cond": [ { "$eq": [ "$B_EU_FUNDS", "Y" ] }, 1, "null"]
                                                },
                                 "bEuFundsNoCount":{
                                                # Set to VALUE_EURO if B_EU_FUNDS = Y
                                            "$cond": [ { "$eq": [ "$B_EU_FUNDS", "N" ] }, 1, "null"]
                                                },
                                "bEuFundsYes": {
                                                # Set to VALUE_EURO if B_EU_FUNDS = Y
                                            "$cond": [ { "$eq": [ "$B_EU_FUNDS", "Y" ] }, "$VALUE_EURO", "null"]
                                                },
                                "bEuFundsNo":  {
                                                # Set to VALUE_EURO if B_EU_FUNDS = N
                                                "$cond": [ { "$eq": [ "$B_EU_FUNDS", "N" ] }, "$VALUE_EURO", "null"]
                                                }
                                }
                },
                {"$group": {"_id": {"cpv_division":"$CPV_DIVISION","year":"$YEAR","country":"$ISO_COUNTRY_CODE"},
                                "sum": {"$sum": "$VALUE_EURO"},
                                "countBEuFundsYes":{"$sum":"$bEuFundsYesCount"},
                                "countBEuFundsNo":{"$sum":"$bEuFundsNoCount"},
                                "count":{"$sum":1},
                                "countOffers": {"$sum": "$NUMBER_OFFERS"},
                                "sumValueBEuFundsYes": { "$sum": "$bEuFundsYes" },
                                "sumValueBEuFundsNo": { "$sum": "$bEuFundsNo" },
                            }
                    },
                    # $merge to write the results out to another collection
                    { "$merge": { "into": "cpv_euro_avg" } } ]

"""
pipeline_q8 solves question 8.
Creates a collection calleed cpv_bucket. Contain pre-prepared data to be able to apply a bucket.
"""
pipeline_q8 = [
    # CAE_NAME must be string, the others must be not null
    {"$match": {"CPV_DIVISION": {"$type": 2},
                "VALUE_EURO": {"$ne": "null", "$lte": 100000000},
                "YEAR": {"$ne": "null"},
                "ISO_COUNTRY_CODE": {"$ne": "null"}}
     },
    # project only the necessary fields
    {"$project": {"CPV_DIVISION": 1,
                  "VALUE_EURO": 1,
                  "ISO_COUNTRY_CODE": 1,
                  "YEAR": 1}},

    # $merge to write the results out to another collection
    {"$merge": {"into": "cpv_bucket"}}]

"""
pipeline_q9 answers q9
Creates collection cpv_time_diff_2, which contains the calculations to obtain the differences for value and time (in days).
"""
pipeline_q9 = [
                # CAE_NAME must be string, the others must be not null
                {"$match": {"CPV_DIVISION": {"$type":2},
                            "VALUE_EURO": {"$ne":"null"},
                            "VALUE_EURO": {"$ne":"null","$lte" : 100000000},
                            "DT_DISPATCH":{"$ne":"null"},
                            "DT_AWARD":{"$ne":"null"},
                            "ISO_COUNTRY_CODE": {"$ne":"null"} }
                },
                # project only the necessary fields
                {"$project": {"CPV_DIVISION": 1,
                                "VALUE_EURO": 1,
                                "ISO_COUNTRY_CODE": 1,
                                "awardDate":{"$dateFromString":{"dateString":"$DT_AWARD"}},
                                "dispatchDate":{"$dateFromString":{"dateString":"$DT_DISPATCH"}},
                                "value":{"$subtract":["$AWARD_VALUE_EURO","$VALUE_EURO"]},
                                "YEAR": 1}},
                {"$project": {"CPV": "$CPV_DIVISION","ISO_COUNTRY_CODE":"$ISO_COUNTRY_CODE","YEAR":"$YEAR",
                               "time": {"$subtract": ["$dispatchDate", "$awardDate"]}, "value": "$value"}},
                {"$addFields": { "time": {"$divide": ["$time", 8.64e+7]}}},
                    # $merge to write the results out to another collection
                    { "$merge": { "into": "cpv_time_diff_2" } } ]

"""
pipeline_companies_10_to_19
(useful in questions 10 to 19)

The query below returns a view with the average VALUE_EURO for each combination
of CAE_NAME, YEAR and ISO_COUNTRY_CODE
"""

pipeline_companies_10_to_19 = [
                # CAE_NAME must be string, the others must be not null
                {"$match": {"CAE_NAME": {"$type":2},
                            'CAE_ADDRESS': {"$type":2},
                            'CAE_TOWN': {"$type":2},
                            "VALUE_EURO": {"$gte":1000, "$lte":100000000},
                            "YEAR": {"$ne":"null"},
                            "ISO_COUNTRY_CODE": {"$ne":"null"} }
                },
                # project only the necessary fields
                {"$project": {"CAE_NAME": 1,
                                "ADDRESS": { "$concat": [ "$CAE_ADDRESS", " ", "$CAE_TOWN" ] },
                                "CAE_ADDRESS": 1,
                                "CAE_TOWN": 1,
                                "VALUE_EURO": 1,
                                "ISO_COUNTRY_CODE": 1,
                                "YEAR": 1,
                                "NUMBER_OFFERS": 1,
                                "bEuFundsYes": {
                                                # Set to VALUE_EURO if B_EU_FUNDS = Y
                                            "$cond": [ { "$eq": [ "$B_EU_FUNDS", "Y" ] }, "$VALUE_EURO", "null"]
                                                },
                                "bEuFundsNo":  {
                                                # Set to VALUE_EURO if B_EU_FUNDS = N
                                                "$cond": [ { "$eq": [ "$B_EU_FUNDS", "N" ] }, "$VALUE_EURO", "null"]
                                                }
                                }
                },
                {"$group": {"_id": {
                                "business": "$CAE_NAME",
                                "address": "$ADDRESS",
                                "year": "$YEAR",
                                "country": "$ISO_COUNTRY_CODE"
                                    },
                                "avg": {"$avg": "$VALUE_EURO"},
                                "sum": {"$sum": "$VALUE_EURO"},
                                "count":{"$sum":1},
                                "sum_no": {"$sum": "$NUMBER_OFFERS"},
                                "sumValueBEuFundsYes": { "$sum": "$bEuFundsYes" },
                                "sumValueBEuFundsNo": { "$sum": "$bEuFundsNo" },
                            }
                    },
                    # $merge to write the results out to another collection
                    { "$merge": { "into": "business_euro_sum" } } ]

"""
pipeline_co_occurrences_20~
(useful in question 20)

The query below returns a view with count of contracts for each combination
of CAE_NAME and WIN_NAME, year and country. The _id contains 'co_name' which is an aggregation of
CAE_NAME and WIN_NAME, year and country of the contracts.

In queries.py, after filtering by year and country, this collection will be used
to obtain the sum of contracts for each pair (co_name).
"""

pipeline_co_occurrences_20 = [
            {"$match": {"CAE_NAME": {"$type":2},
                        'WIN_NAME': {"$type":2},
                        "VALUE_EURO": {"$gte":1000, "$lte":100000000},
                        "YEAR": {"$ne":"null"},
                        "ISO_COUNTRY_CODE": {"$ne":"null"}
                       }
            },
            {"$project": {"CAE_NAME": 1,
                          "WIN_NAME": 1,
                          "VALUE_EURO": 1,
                          "ISO_COUNTRY_CODE": 1,
                          "YEAR": 1,
                          "CO_NAME": { "$concat": [ "$CAE_NAME", " with ", "$WIN_NAME" ] },
                         }
            },
            {"$group": {"_id": {"co_name": "$CO_NAME",
                                "year": "$YEAR",
                                "country": "$ISO_COUNTRY_CODE",
                               },
                        "count":{"$sum":1},
                       }
            },
            { "$sort" : { "count" : -1} },
            { "$merge": { "into": "business_co_occurrences" } }                       
           ]